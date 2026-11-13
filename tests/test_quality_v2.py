"""Caminho v2 do motor Qualidade (quality_judge="v2"): âncora B, silêncio, confiança do A, vizinhos, triagem e refino independente."""
import numpy as np

from transcritor import quality_engine as qe
from transcritor.audio import Chunk
from transcritor.config import Config
from transcritor.ollama_client import OllamaError, build_refine_prompt, build_review_v2_prompt, build_triage_prompt

from .test_quality_loop import Job, ScriptedClient, Session, make_run


class V2Client(ScriptedClient):
    def __init__(self, a_texts, reviews, triage=None, fail_review=False):
        super().__init__(a_texts, [])
        self.v2_reviews, self.v2_calls, self.triage_calls, self.triage_result, self.fail_review = list(reviews), [], [], triage, fail_review

    def review_v2(self, model, wav, a, b, diffs, seconds, context="", before="", after="", notes=False, trusted=False):
        self.v2_calls.append({"a": a, "b": b, "diffs": diffs, "context": context, "before": before, "after": after, "notes": notes,
                              "trusted": trusted})
        if self.fail_review:
            raise OllamaError("x")
        return self.v2_reviews.pop(0)

    def triage(self, model, items, context="", limit=20):
        self.triage_calls.append(items)
        return self.triage_result

    def transcribe_reason(self, model, wav, context="", points=None, critique="", terms=None):
        self.reason_calls = getattr(self, "reason_calls", []) + [{"points": points, "critique": critique, "terms": terms}]
        return self.a_texts.pop(0)


def rv(picks=(), speech=True, fixes=(), conf=0.9, **extra):
    return {"analysis": "ok", "speech": speech, "picks": list(picks), "fixes": list(fixes), "confidence": conf, **extra}


def run_v2(tmp_path, monkeypatch, client, a, b, loops=0, **cfg):
    return make_run(tmp_path, monkeypatch, client, Session(list(b) + ["x"] * 20), a, loops=loops, quality_judge="v2", **cfg)


def test_untrusted_a_never_fills_the_silence_of_b(tmp_path, monkeypatch):
    a = ["um dois tres", "texto inventado de musica aqui", "quatro cinco", "mais inventado algo ali agora", "seis sete"]
    b = ["um dois tres", "", "quatro cinco", "", "seis sete"]
    client = V2Client([], [rv(speech=True), rv(speech=True)])  # até com o revisor dizendo "fala", o A não é confiável neste arquivo
    (texts, _f, _e, rep), job = run_v2(tmp_path, monkeypatch, client, a, b)
    assert texts == b and rep.judge == "v2" and rep.silenced == 2
    assert rep.signals["a_trusted"] is False and rep.signals["only_a_rate"] == 0.4
    assert any("pouco confiável" in w for w in job.warnings)
    assert len(client.v2_calls) == 2 and client.v2_calls[0]["b"] == ""  # o revisor ouviu os dois silêncios (pergunta de fala)


def test_trusted_a_applies_the_reviewers_pick_per_divergence(tmp_path, monkeypatch):
    a, b = ["the cat sat on a mat", "all good here"], ["the cat sat on the mat", "all good here"]
    client = V2Client([], [rv(picks=["A"])])
    (texts, _f, _e, rep), _ = run_v2(tmp_path, monkeypatch, client, a, b)
    assert texts == ["the cat sat on a mat", "all good here"] and rep.signals["a_trusted"] is True and rep.merged == 1
    call = client.v2_calls[0]
    assert call["diffs"] == [{"a": "a", "b": "the"}] and call["after"] == "all good here" and call["before"] == ""  # vizinhos como contexto


def test_reviewer_fixes_a_word_both_got_wrong_and_b_is_the_default(tmp_path, monkeypatch):
    a, b = ["Segundo o Doutor Mall, alguns pacientes"], ["Segundo o Dr. Mall, alguns pacientes"]
    client = V2Client([], [rv(picks=["B"], fixes=[{"from": "Mall", "to": "Moll"}])])
    (texts, _f, _e, rep), _ = run_v2(tmp_path, monkeypatch, client, a, b)
    assert texts == ["Segundo o Dr. Moll, alguns pacientes"] and rep.from_b == 1
    client = V2Client([], [], fail_review=True)  # revisão falhou: vale o B
    (texts, _f, _e, _r), job = run_v2(tmp_path, monkeypatch, client, a, b)
    assert texts == b and any("sem resposta válida" in w for w in job.warnings)


def test_too_many_differences_still_review_b_with_no_picks(tmp_path, monkeypatch):
    a = [" ".join(f"s a{i}" for i in range(12))]  # 12 pontos de divergência separados por palavras iguais
    b = [" ".join(f"s b{i}" for i in range(12))]
    client = V2Client([], [rv(picks=["A"] * 12)])
    (texts, _f, _e, _r), _ = run_v2(tmp_path, monkeypatch, client, a, b)
    assert texts == b and client.v2_calls[0]["diffs"] is None  # diverge demais: o B fica; o revisor só confere o B


def test_refinement_is_independent_and_only_for_fragments_the_reviewer_changed(tmp_path, monkeypatch):
    a, b = ["the cat sat on a mat", "all good here"], ["the cat sat on the mat", "all good here"]
    r1 = rv(picks=["A"], critique="x", terms=["Mat"])
    r2 = rv(picks=["B"])
    client = V2Client(["the cat sat on a mat"], [r1, r2])  # A' com feedback: ainda difere do B'
    sess = Session(["the cat sat on the mat", "all good here", "the cat sat on the mat"])
    (texts, _f, _e, rep), _ = make_run(tmp_path, monkeypatch, client, sess, a, loops=1, quality_judge="v2")
    assert rep.loops_done == 1 and [p["targets"] for p in rep.passes] == [2, 1]  # só o fragmento que o revisor mexeu
    assert texts[0] == "the cat sat on the mat"  # 2ª opinião independente: agora o revisor ficou com o B
    # independência: A e B recebem só os termos, NUNCA o texto escolhido para aquele fragmento
    assert client.feedbacks and "the cat sat on a mat" not in client.feedbacks[0] and "Mat" in client.feedbacks[0]
    assert "Mat" in sess.prompts[-1] and "cat" not in sess.prompts[-1]
    assert set(rep.texts) >= {"pass1", "pass2", "final", "a1", "b1"}


def test_triage_adds_suspects_and_glossary_to_the_review(tmp_path, monkeypatch):
    a = b = ["a casa grande", "Odiseu chegou", "fim da frase"]
    client = V2Client([], [rv()], triage={"analysis": "x", "glossary": ["Odysseus"], "suspects": [2, 99, "x"]})
    (texts, _f, _e, rep), _ = run_v2(tmp_path, monkeypatch, client, a, b, quality_triage=True)
    assert len(client.triage_calls) == 1 and [it["n"] for it in client.triage_calls[0]] == [1, 2, 3]
    assert rep.triage["suspects"] == 1 and rep.reviewed == 1  # só o fragmento 2 foi ouvido (A e B concordavam, o gate não marcaria)
    assert "Odysseus" in client.v2_calls[0]["context"]


def test_v1_is_unchanged_by_default(tmp_path, monkeypatch):
    assert Config().quality_judge == "v1" and Config().quality_triage is False


def test_v2_prompts_render_from_files():
    p = build_review_v2_prompt("the cat sat on a mat", "the cat sat on the mat", [{"a": "a", "b": "the"}], 11.6, "Mat", "antes", "depois", True)
    assert "1. A: \"a\" | B: \"the\"" in p and "about 12 seconds" in p and "before: \"antes\"" in p and "- critique:" in p and "Mat" in p
    assert p.index("analysis") < p.index("picks") < p.index("fixes")  # raciocínio antes das decisões
    s = build_review_v2_prompt("texto", "", [], 8.0)
    assert "heard NO speech" in s and "texto" in s
    assert "differ too much to list" in build_review_v2_prompt("a", "b", None, 5.0)
    t = build_triage_prompt([{"n": 1, "B": "x", "A": "y"}], "Odysseus", 5)
    assert "1 numbered fragments" in t and "at most 5" in t and '"n": 1' in t and "Odysseus" in t


def _refine_run(tmp_path, monkeypatch, level):
    a, b = ["the cat sat on a mat", "all good here"], ["the cat sat on the mat", "all good here"]
    client = V2Client(["the cat sat on the mat"], [rv(picks=["A"], critique="A said a, it is the", terms=["Mat"]), rv(picks=["B"])])
    sess = Session(["the cat sat on the mat", "all good here", "the cat sat on the mat"])
    out = make_run(tmp_path, monkeypatch, client, sess, a, loops=1, quality_judge="v2", quality_feedback=level)
    return client, out[0]


def test_feedback_levels_points_and_critique_use_reasoning_generation(tmp_path, monkeypatch):
    client, (texts, _f, _e, rep) = _refine_run(tmp_path, monkeypatch, "points")
    call = client.reason_calls[0]
    assert call["points"] == [{"a": "a", "b": "the"}] and call["critique"] == "" and call["terms"] == ["Mat"]  # pontos SEM veredito, sem crítica
    assert client.feedbacks == [] and rep.loops_done == 1 and texts[0] == "the cat sat on the mat"
    client, _ = _refine_run(tmp_path, monkeypatch, "critique")
    assert client.reason_calls[0]["critique"] == "A said a, it is the"  # só neste nível a crítica do revisor chega ao gerador


def test_refine_prompt_carries_points_without_verdict_and_asks_for_reasoning():
    p = build_refine_prompt([{"a": "a", "b": "the"}, {"a": "today", "b": ""}], "A misheard it", ["Mat"], "ctx")
    assert '"a" or "the"' in p and '"today" or "(nothing)"' in p and "A misheard it" in p and "Mat" in p
    assert p.index("analysis") < p.index("transcription")
    assert "Points of doubt" not in build_refine_prompt(None, "", None)


def test_review_prompt_tone_follows_the_trust_in_a():
    anchor = build_review_v2_prompt("a b", "a c", [{"a": "b", "b": "c"}], 5.0)
    neutral = build_review_v2_prompt("a b", "a c", [{"a": "b", "b": "c"}], 5.0, trusted=True)
    assert "B is the reference" in anchor and "B is the reference" not in neutral
    assert "never from which system" in neutral and "never from which system" not in anchor


def test_engine_asks_the_reviewer_neutrally_only_when_a_is_trusted(tmp_path, monkeypatch):
    a, b = ["the cat sat on a mat"], ["the cat sat on the mat"]
    client = V2Client([], [rv(picks=["B"])])
    run_v2(tmp_path, monkeypatch, client, a, b)
    assert client.v2_calls[0]["trusted"] is True  # A confiável (razão A/B ~1, sem "só A")
    client = V2Client([], [rv(speech=False)])
    run_v2(tmp_path, monkeypatch, client, ["fala um", "inventado de musica aqui"], ["fala um", ""])
    assert client.v2_calls[0]["trusted"] is False  # A pouco confiável: âncora no B


def test_json_agents_cap_the_response_length(monkeypatch):
    """Sem limite de tokens, um modelo que não fecha o campo `analysis` gera até o tempo limite (15 min por chamada)."""
    from transcritor import ollama_client as oc

    seen = []
    client = oc.OllamaClient("http://x")
    monkeypatch.setattr(client, "_chat", lambda payload: seen.append(payload["options"]) or "{}")
    for call in (lambda: client.review_self("m", b"", "t", 5.0), lambda: client.review_v2("m", b"", "a", "b", [], 5.0),
                 lambda: client.transcribe_self("m", b"", "", "x", "y"), lambda: client.transcribe_reason("m", b""),
                 lambda: client.review("m", b"", "a", "b"), lambda: client.triage("m", [{"n": 1, "B": "x", "A": "y"}])):
        try:
            call()
        except oc.OllamaError:
            pass  # a resposta "{}" é inválida de propósito: só importa o que foi enviado
    assert len(seen) == 6 and all("num_predict" in o for o in seen)
    assert seen[0]["num_predict"] == oc.MAX_JSON_TOKENS and seen[-1]["num_predict"] == oc.MAX_TRIAGE_TOKENS
