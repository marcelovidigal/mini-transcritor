"""Modo de qualidade 1: autorrefinamento gerador <-> revisor, sem WhisperX (dublês, sem GPU)."""
from transcritor.ollama_client import OllamaError, build_self_refine_prompt, build_self_review_prompt

from .test_quality_loop import ScriptedClient, Session, make_run


class SelfClient(ScriptedClient):
    def __init__(self, reviews, regen=(), fail_review=False):
        super().__init__([], [])
        self.self_reviews, self.regen, self.review_calls_self, self.regen_calls, self.fail_review = list(reviews), list(regen), [], [], fail_review

    def review_self(self, model, wav, text, seconds, context="", before="", after=""):
        self.review_calls_self.append({"model": model, "text": text, "before": before, "after": after})
        if self.fail_review:
            raise OllamaError("x")
        return self.self_reviews.pop(0)

    def transcribe_self(self, model, wav, context="", previous="", critique="", terms=None):
        self.regen_calls.append({"model": model, "previous": previous, "critique": critique, "terms": terms})
        return self.regen.pop(0)


def rs(verdict="ok", speech=True, fixes=(), critique="", terms=(), conf=0.9):
    return {"analysis": "x", "speech": speech, "verdict": verdict, "fixes": list(fixes), "critique": critique, "terms": list(terms), "confidence": conf}


def run_self(tmp_path, monkeypatch, client, a, loops=2, **cfg):
    return make_run(tmp_path, monkeypatch, client, Session([]), a, loops=loops, quality_self=True, **cfg)


def test_ok_verdict_stops_the_loop_and_keeps_the_text(tmp_path, monkeypatch):
    client = SelfClient([rs("ok")])
    (texts, _f, _e, rep), job = run_self(tmp_path, monkeypatch, client, ["the cat sat on the mat"])
    assert texts == ["the cat sat on the mat"] and rep.judge == "self" and rep.loops_done == 0 and client.regen_calls == []
    assert rep.b_source == "" and len(rep.passes) == 1 and set(rep.texts) >= {"a1", "pass1", "final"}


def test_fix_verdict_sends_previous_text_and_critique_back_to_the_generator(tmp_path, monkeypatch):
    client = SelfClient([rs("fix", critique="the last word is mat, not map", terms=["Mat"]), rs("ok")], regen=["the cat sat on the mat"])
    (texts, _f, _e, rep), _ = run_self(tmp_path, monkeypatch, client, ["the cat sat on the map"])
    call = client.regen_calls[0]
    assert call["previous"] == "the cat sat on the map" and "mat, not map" in call["critique"] and call["terms"] == ["Mat"]
    assert texts == ["the cat sat on the mat"] and rep.loops_done == 1 and [p["targets"] for p in rep.passes] == [1, 1]
    assert client.review_calls_self[1]["text"] == "the cat sat on the mat"  # a 2ª revisão julga o texto refinado


def test_two_refinement_rounds_and_per_pass_texts(tmp_path, monkeypatch):
    client = SelfClient([rs("fix", critique="a"), rs("fix", critique="b"), rs("ok")], regen=["one two three four", "one two three five"])
    (texts, _f, _e, rep), _ = run_self(tmp_path, monkeypatch, client, ["one two three"])
    assert rep.loops_done == 2 and [p["pass"] for p in rep.passes] == [1, 2, 3]
    assert texts == ["one two three five"] and set(rep.texts) >= {"pass1", "pass2", "pass3"}
    assert client.regen_calls[1]["previous"] == "one two three four"  # o refino parte da tentativa anterior


def test_last_round_has_no_refinement_but_still_applies_the_reviewers_fixes(tmp_path, monkeypatch):
    client = SelfClient([rs("fix", fixes=[{"from": "Mall", "to": "Moll"}])])
    (texts, _f, _e, rep), _ = run_self(tmp_path, monkeypatch, client, ["Segundo o Dr. Mall, alguns pacientes"], loops=0)
    assert texts == ["Segundo o Dr. Moll, alguns pacientes"] and client.regen_calls == []


def test_no_speech_becomes_silence_and_low_confidence_is_ignored(tmp_path, monkeypatch):
    client = SelfClient([rs("fix", speech=False), rs("fix", speech=False, conf=0.2)])
    (texts, _f, _e, rep), _ = run_self(tmp_path, monkeypatch, client, ["texto inventado de musica", "fala de verdade aqui"], loops=0)
    assert texts == ["", "fala de verdade aqui"] and rep.silenced == 1


def test_generator_cannot_rewrite_the_fragment_beyond_the_limit(tmp_path, monkeypatch):
    client = SelfClient([rs("fix", critique="x"), rs("ok")], regen=["completamente outra frase sem relacao nenhuma"])
    (texts, _f, _e, _r), _ = run_self(tmp_path, monkeypatch, client, ["a casa grande e bonita"])
    assert texts == ["a casa grande e bonita"]  # reescrita excessiva: vale a tentativa anterior


def test_review_failure_keeps_the_text_and_empty_fragments_are_skipped(tmp_path, monkeypatch):
    client = SelfClient([], fail_review=True)
    (texts, _f, _e, rep), job = run_self(tmp_path, monkeypatch, client, ["fala um", ""])
    assert texts == ["fala um", ""] and rep.reviewed == 1 and any("sem resposta válida" in w for w in job.warnings)


def test_neighbours_reach_the_self_reviewer(tmp_path, monkeypatch):
    client = SelfClient([rs(), rs()])
    run_self(tmp_path, monkeypatch, client, ["primeira fala", "segunda fala"], loops=0)
    assert client.review_calls_self[0]["after"] == "segunda fala" and client.review_calls_self[1]["before"] == "primeira fala"


def test_self_prompts_render_from_files():
    p = build_self_review_prompt("olá mundo", 8.2, "Ctx", "antes", "depois")
    assert "Transcription: olá mundo" in p and "about 8 seconds" in p and p.index("analysis") < p.index("verdict") < p.index("fixes")
    q = build_self_refine_prompt("olá mundo", "faltou uma palavra", ["Moll"], "")
    assert '"olá mundo"' in q and "faltou uma palavra." in q and "Moll" in q and q.index("- analysis") < q.index("- transcription")
