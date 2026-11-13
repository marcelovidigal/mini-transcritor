"""Modo de qualidade 1 v2: porteiro de fala (Silero) e votação entre hipóteses do gerador (dublês, sem GPU)."""
import numpy as np

from transcritor import quality_engine, speechgate, vote
from transcritor.ollama_client import build_self_review_prompt, build_variant_prompt

from .test_quality_self import SelfClient, rs, run_self


# --- votação: consenso, dúvidas, estabilidade -------------------------------------------------------------------------------------------

def test_majority_swap_wins_and_a_lone_disagreement_becomes_a_doubt():
    c = vote.consensus(["the cat sat on the map today", "the cat sat on the mat today", "the cat sat on the mat today"])
    assert c["text"] == "the cat sat on the mat today" and c["applied"] == 1 and c["doubts"] == 0
    c = vote.consensus(["the cat sat on the mat today", "the cat sat on the mat today", "the dog sat on the mat today"])
    assert c["text"] == "the cat sat on the mat today" and c["doubts"] == 1
    assert vote.doubts_of(c["text"], c["alt"]) == [{"current": "cat", "other": "dog"}]


def test_choices_are_composed_by_code():
    c = vote.consensus(["um dois tres quatro cinco", "um dois tres quarto cinco", "um doze tres quatro cinco"])
    assert c["text"] == "um dois tres quatro cinco" and c["doubts"] == 2
    d = vote.doubts_of(c["text"], c["alt"])
    assert [x["other"] for x in d] == ["doze", "quarto"] or [x["other"] for x in d] == ["quarto", "doze"]
    first_other = next(k for k, x in enumerate(d) if x["other"] == "doze")
    choices = ["current"] * len(d)
    choices[first_other] = "other"
    assert vote.apply_choices(c["text"], c["alt"], choices) == "um doze tres quatro cinco"
    assert vote.apply_choices(c["text"], c["alt"], ["current"] * len(d)) == c["text"]


def test_most_hypotheses_empty_means_silence_and_empty_variant_failure_keeps_text():
    assert vote.consensus(["texto inventado", "", ""])["text"] == ""
    assert vote.consensus(["texto inventado", "texto inventado", ""])["text"] == "texto inventado"
    assert vote.consensus(["texto inventado", ""])["text"] == "texto inventado"  # só 2 hipóteses: uma vazia não é maioria


def test_wildly_different_hypotheses_are_unstable():
    c = vote.consensus(["a b c d e f", "x y z w v u", "q r s t o p"])
    assert c["stability"] < vote.UNSTABLE_BELOW
    assert vote.consensus(["a b c", "a b c", "a b c"])["stability"] == 1.0


# --- prompts ---------------------------------------------------------------------------------------------------------------------------------

def test_review_prompt_lists_the_doubts_only_when_there_are_some():
    plain = build_self_review_prompt("a b c", 5)
    assert "consensus" not in plain and "picks" not in plain
    p = build_self_review_prompt("a b c", 5, doubts=[{"current": "cat", "other": "dog"}], unstable=True)
    assert '1. current: "cat" | other: "dog"' in p and "picks" in p and "no clear speech" in p


def test_variant_prompts_are_distinct():
    assert build_variant_prompt("literal") != build_variant_prompt("careful")
    assert "word by word" in build_variant_prompt("literal") and "numbers, names" in build_variant_prompt("careful")


# --- no motor --------------------------------------------------------------------------------------------------------------------------------

class VoteClient(SelfClient):
    def __init__(self, reviews, variants, **kw):
        super().__init__(reviews, **kw)
        self.variants, self.variant_calls, self.review_args = dict(variants), [], []

    def transcribe_variant(self, model, wav, context="", kind="literal"):
        self.variant_calls.append(kind)
        return self.variants[kind].pop(0)

    def review_self(self, model, wav, text, seconds, context="", before="", after="", doubts=None, unstable=False):
        self.review_args.append({"text": text, "doubts": doubts, "unstable": unstable})
        return super().review_self(model, wav, text, seconds, context, before, after)


def test_vote_consensus_goes_to_the_reviewer_and_its_picks_are_applied(tmp_path, monkeypatch):
    client = VoteClient([rs("ok")], {"literal": ["a casa grande e bonita"], "careful": ["a casa grande e bonita"]})
    client.self_reviews = [{**rs("ok"), "picks": ["other"]}]
    (texts, _f, _e, rep), _ = run_self(tmp_path, monkeypatch, client, ["a casa grade e bonita"], loops=0, quality_vote=True)
    # as duas variantes concordam entre si: troca por maioria, sem dúvida
    assert texts == ["a casa grande e bonita"] and rep.vote["applied"] == 1 and rep.vote["with_doubts"] == 0
    assert client.variant_calls == ["literal", "careful"] and rep.texts["consenso"] == ["a casa grande e bonita"]
    assert client.review_args[0]["doubts"] is None and "votacao" in rep.seconds


def test_doubt_is_sent_to_the_reviewer_and_applied_by_the_code(tmp_path, monkeypatch):
    client = VoteClient([], {"literal": ["a casa grande e bonita"], "careful": ["a casa grande e bonita"]})
    client.variants = {"literal": ["a casa branca e bonita"], "careful": ["a casa grande e bonita"]}
    client.self_reviews = [{**rs("ok"), "picks": ["other"]}]
    (texts, _f, _e, rep), _ = run_self(tmp_path, monkeypatch, client, ["a casa grande e bonita"], loops=0, quality_vote=True)
    assert client.review_args[0]["doubts"] == [{"current": "grande", "other": "branca"}]
    assert texts == ["a casa branca e bonita"] and rep.vote["doubts"] == 1


def test_vote_off_keeps_the_old_flow(tmp_path, monkeypatch):
    client = VoteClient([rs("ok")], {"literal": [], "careful": []})
    (texts, _f, _e, rep), _ = run_self(tmp_path, monkeypatch, client, ["a casa grande"], loops=0)
    assert client.variant_calls == [] and rep.vote == {} and texts == ["a casa grande"]


def test_voted_silence_skips_the_reviewer(tmp_path, monkeypatch):
    client = VoteClient([], {"literal": [""], "careful": [""]})
    (texts, _f, _e, rep), _ = run_self(tmp_path, monkeypatch, client, ["musica inventada aqui"], loops=0, quality_vote=True)
    assert texts == [""] and rep.vote["silenced"] == 1 and client.review_args == []


# --- porteiro ----------------------------------------------------------------------------------------------------------------------------------

def test_gate_discards_fragments_before_generation(tmp_path, monkeypatch):
    from transcritor.audio import Chunk
    from transcritor.config import Config
    from .test_quality_loop import Job

    ratios = iter([0.0, 0.9, 0.9])
    monkeypatch.setattr(speechgate, "available", lambda: True)
    monkeypatch.setattr(speechgate, "speech_ratio", lambda s: next(ratios))
    monkeypatch.delenv("MT_KEEP_WORK", raising=False)
    generated = []

    def gen(job, client, chunks, config, upd):  # como o _transcribe_all: um texto por fragmento RECEBIDO
        generated.append(len(chunks))
        return [f"fala {k}" for k in range(len(chunks))], 0, ""

    chunks = [Chunk(i * 10.0, i * 10.0 + 8, np.zeros(16000, dtype=np.int16)) for i in range(3)]
    cfg = Config(work_dir=str(tmp_path), quality_self=True, quality_loops=0, quality_reviewer="fake-r", quality_gate=True)
    texts, _f, _e, rep = quality_engine.run(Job(), SelfClient([rs("ok"), rs("ok")]), cfg, None, chunks, lambda p, s=None: None, gen)
    assert generated == [2] and rep.gate["discarded"] == 1 and texts == ["", "fala 0", "fala 1"]


def test_gate_that_finds_no_speech_anywhere_transcribes_everything(tmp_path, monkeypatch):
    monkeypatch.setattr(speechgate, "available", lambda: True)
    monkeypatch.setattr(speechgate, "speech_ratio", lambda s: 0.0)
    client = SelfClient([rs("ok"), rs("ok")])
    (texts, _f, _e, rep), job = run_self(tmp_path, monkeypatch, client, ["a b", "c d"], loops=0, quality_gate=True)
    assert texts == ["a b", "c d"] and rep.gate["discarded"] == 0 and any("nenhum fragmento" in w for w in job.warnings)


def test_gate_off_or_unavailable_lets_everything_through(tmp_path, monkeypatch):
    monkeypatch.setattr(speechgate, "available", lambda: False)
    client = SelfClient([rs("ok"), rs("ok")])
    (texts, _f, _e, rep), job = run_self(tmp_path, monkeypatch, client, ["a b", "c d"], loops=0, quality_gate=True)
    assert texts == ["a b", "c d"] and rep.gate == {} and any("Porteiro" in w for w in job.warnings)


def test_speech_ratio_is_one_when_the_detector_is_missing(monkeypatch):
    monkeypatch.setattr(speechgate, "available", lambda: False)
    assert speechgate.speech_ratio(np.zeros(16000, dtype=np.int16)) == 1.0 and speechgate.has_speech(np.zeros(10, dtype=np.int16))


def test_variant_generation_is_capped_in_tokens(monkeypatch):
    """Sem limite, o modelo pequeno repete texto em música até estourar o tempo (as hipóteses da votação rodam em fragmentos sem fala)."""
    from transcritor import ollama_client as oc

    seen = []
    client = oc.OllamaClient("http://x")
    monkeypatch.setattr(client, "_chat", lambda payload: seen.append(payload["options"]) or "texto")
    assert client.transcribe_variant("m", b"", "", "literal") == "texto"
    assert seen[0]["num_predict"] == oc.MAX_VARIANT_TOKENS
