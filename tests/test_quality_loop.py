"""Laço de refinamento, especialistas e os dois modos de qualidade (com dublês: sem GPU, sem Ollama)."""
import shutil

import numpy as np
import pytest

from transcritor import engines, pipeline, specialists, workdir
from transcritor import quality_engine as qe
from transcritor.audio import Chunk
from transcritor.config import Config
from transcritor.engines import Engine
from transcritor.ollama_client import OllamaError, build_feedback_note, build_review_prompt

from .test_api import start, wait_done


class Job:
    id, model, language, context, engine, owner, report = "j1", "fake-a", "en", "", "quality", None, None

    def __init__(self):
        self.warnings = []

    def check_cancel(self):
        pass


class ScriptedClient:
    """A (transcribe) e o revisor devolvem o que o teste combinar; guarda o que recebeu."""

    def __init__(self, a_texts, reviews):
        self.a_texts, self.reviews, self.feedbacks, self.review_calls = list(a_texts), list(reviews), [], []

    def audio_models(self):
        return [{"name": "fake-r", "size": 1}]

    def transcribe(self, model, wav, context="", feedback=""):
        self.feedbacks.append(feedback)
        return self.a_texts.pop(0)

    def review(self, model, wav, a, b, context="", c="", notes=False, c_is_previous=False):
        self.review_calls.append({"a": a, "b": b, "c": c, "notes": notes, "prev": c_is_previous})
        return self.reviews.pop(0)

    def polish(self, model, texts, language="", glossary=""):
        return texts

    def unload_all(self):
        return 0


class Session:
    def __init__(self, b_texts):
        self.b_texts, self.prompts, self.closed = list(b_texts), [], False

    def transcribe(self, samples, prompt=""):
        self.prompts.append(prompt)
        return self.b_texts.pop(0)

    def close(self):
        self.closed = True


def make_run(tmp_path, monkeypatch, client, session, a1, loops=2, reviewer="fake-r", **cfg):
    class Eng(Engine):
        id, label, module, pip = "fake-wx", "FakeWX", "os", "x"

        def catalog(self):
            return [{"name": "m", "approx_size": "", "note": ""}]

        def is_installed(self, model):
            return True

        def session(self, model, *, device="auto", context="", job=None):
            return session

    monkeypatch.setitem(engines.ENGINES, "fake-wx", Eng())
    monkeypatch.delenv("MT_KEEP_WORK", raising=False)
    config = Config(work_dir=str(tmp_path), quality_b_engine="fake-wx", quality_b_model="m", quality_loops=loops,
                    quality_reviewer=reviewer, quality_prefer="B", **cfg)
    chunks = [Chunk(i * 10.0, i * 10.0 + 8, np.zeros(16000, dtype=np.int16)) for i in range(len(a1))]
    job = Job()
    return qe.run(job, client, config, None, chunks, lambda p, s=None: None, lambda *x: (list(a1), 0, "")), job


def test_loop_stops_early_when_nothing_changes(tmp_path, monkeypatch):
    client = ScriptedClient(["the cat sat down"], [{"verdict": "B", "text": "", "confidence": 0.9, "critique": "A heard 'wurld'", "terms": ["World"]}])
    sess = Session(["the cat sat now", "the cat sat now"])  # pass 1, pass 2
    client.a_texts = ["the cat sat now"]  # A concorda depois do feedback
    (texts, failures, _e, rep), _job = make_run(tmp_path, monkeypatch, client, sess, ["the cat sat down"])
    assert texts == ["the cat sat now"] and failures == 0
    assert rep.loops == 2 and rep.loops_done == 1  # a 2ª passada não mudou nada: parou
    assert "wurld" in client.feedbacks[0] and "the cat sat now" in client.feedbacks[0]  # crítica + versão anterior no prompt de A
    assert "World" in sess.prompts[1] and "the cat sat now" in sess.prompts[1]  # termos + versão anterior no initial_prompt de B
    assert client.review_calls[0]["notes"] is True  # com refinamento o revisor devolve crítica e termos
    assert sess.closed
    assert [p["pass"] for p in rep.passes] == [1, 2]


def test_two_refinement_passes_run_and_record_history(tmp_path, monkeypatch):
    r1 = {"verdict": "A", "text": "", "confidence": 0.9, "critique": "x", "terms": []}
    r2 = {"verdict": "A", "text": "", "confidence": 0.9, "critique": "y", "terms": ["Moll"]}
    r3 = {"verdict": "B", "text": "", "confidence": 0.9, "critique": "z", "terms": []}
    client = ScriptedClient(["hello world foo bar", "hello world foo baz"], [r1, r2, r3])
    sess = Session(["hello world foo bar", "hello world foo bar2", "hello world foo baz2"])
    (texts, _f, _e, rep), _ = make_run(tmp_path, monkeypatch, client, sess, ["hello wurld foo"])
    assert rep.loops_done == 2 and len(rep.passes) == 3
    c = [x["c"] for x in client.review_calls]
    assert c[0] == "" and c[1] and c[2]  # nas passadas 2 e 3 a versão final anterior entra como candidato C
    assert [x["notes"] for x in client.review_calls] == [True, True, False]  # a última passada não precisa de notas
    assert [x["prev"] for x in client.review_calls][1:] == [True, True]
    assert set(rep.texts) >= {"pass1", "pass2", "pass3", "final"}
    assert texts == rep.texts["final"] == rep.texts["pass3"]


def test_no_loops_keeps_single_pass_behaviour(tmp_path, monkeypatch):
    client = ScriptedClient([], [{"verdict": "B", "text": "", "confidence": 0.9}])
    sess = Session(["hello world foo bar"])
    (texts, _f, _e, rep), _ = make_run(tmp_path, monkeypatch, client, sess, ["hello wurld foo"], loops=0)
    assert texts == ["hello world foo bar"] and rep.loops_done == 0 and len(rep.passes) == 1
    assert client.feedbacks == [] and client.review_calls[0]["notes"] is False and sess.closed


def test_manifest_has_pass_history(tmp_path, monkeypatch):
    import json
    import os

    client = ScriptedClient(["hello world foo bar"], [{"verdict": "B", "text": "", "confidence": 0.9, "critique": "", "terms": []}])
    sess = Session(["hello world foo bar", "hello world foo bar"])
    cfgp = dict(loops=1)
    (_t, _f, _e, _rep), job = make_run(tmp_path, monkeypatch, client, sess, ["hello wurld foo"], **cfgp)
    d = os.path.join(str(tmp_path), job.id)
    assert os.path.isdir(d)
    m = json.load(open(os.path.join(d, "manifesto.json"), encoding="utf-8"))
    assert m["fragments"][0]["passes"][0]["pass"] == 1 and m["report"]["loops"] == 1
    assert "texts" not in m["report"]


# ---------- especialistas ----------

def test_snap_terms_only_changes_close_words():
    assert specialists.snap_terms("Odyseus chegou a Ítaca.", ["Odysseus", "Ítaca"]) == "Odysseus chegou a Ítaca."
    assert specialists.snap_terms("Penelope, disse ele", ["Penelope"]) == "Penelope, disse ele"
    assert specialists.snap_terms("o barco chegou", ["Odysseus"]) == "o barco chegou"  # nada parecido: intacto
    assert specialists.snap_terms("Telemaco falou.", ["Telemachus"]) == "Telemaco falou."  # parecido, mas abaixo do limiar de 0,84
    assert specialists.snap_terms("Penelopee, disse", ["Penelope"]) == "Penelope, disse"  # mantém a pontuação em volta


def test_build_glossary_merges_context_and_reviewer_terms():
    g = specialists.build_glossary("Odysseus, Ítaca", [{"terms": ["Penelope", "odysseus"]}, None, {"terms": "x"}])
    assert g == ["Odysseus", "Ítaca", "Penelope"]


class PolishClient:
    def __init__(self, out=None, fail=False):
        self.out, self.fail = out, fail

    def polish(self, model, texts, language="", glossary=""):
        if self.fail:
            raise OllamaError("x")
        return self.out(texts) if self.out else texts


def test_polish_accepts_only_punctuation_changes():
    texts = ["ola mundo como vai", "tudo bem por aqui"]
    ok, st = specialists.polish(PolishClient(lambda t: ["Olá mundo, como vai?", "tudo certo por aqui"]), "m", texts, "pt")
    assert ok[0] == "Olá mundo, como vai?" and ok[1] == texts[1]  # a 2ª trocou palavras: descartada
    assert st["changed"] == 1 and st["rejected"] == 1
    same, st = specialists.polish(PolishClient(fail=True), "m", texts, "pt")
    assert same == texts and st["failed_batches"] == 1  # falha do modelo nunca derruba o trabalho


def test_review_translation_guardrails():
    pairs = [("I have 3 sons", "Eu tenho 3 filhos"), ("Hello there", "Olá"), ("Fine", "Tudo bem")]
    out = ["Tenho 3 filhos", "Olá, tudo bem com você, meu caro amigo de longa data?", "Tudo bem"]
    class C:
        def review_translation(self, model, p, language, glossary=""):
            return out

    new, st = specialists.review_translation(C(), "m", pairs, "pt-BR")
    assert new[0] == "Tenho 3 filhos"  # corrigida
    assert new[1] == "Olá" and st["rejected"] == 1  # longa demais: descartada
    assert new[2] == "Tudo bem"
    out[:] = ["Tenho 5 filhos", "Olá", "Tudo bem"]  # mudou o número
    assert specialists.review_translation(C(), "m", pairs, "pt-BR")[0][0] == "Eu tenho 3 filhos"


# ---------- prompts ----------

def test_prompts_carry_feedback_and_notes():
    n = build_feedback_note("hello world", "A misheard a name", ["Moll"])
    assert "hello world" in n and "misheard" in n and "Moll" in n and build_feedback_note("") == ""
    p = build_review_prompt("a", "b", notes=True)
    assert "critique" in p and "terms" in p and "critique" not in build_review_prompt("a", "b")
    assert "previous final" in build_review_prompt("a", "b", c="c", c_is_previous=True)


# ---------- os dois modos no app ----------

def test_quality_1_is_the_self_refinement_mode_with_fixed_models():
    cfg = qe.apply_mode("quality-1", Config())
    assert (cfg.quality_self, cfg.quality_reviewer, cfg.quality_loops, cfg.quality_specialists) == (True, "gemma4:12b", 1, False)
    assert qe.mode_generator("quality-1") == "gemma4:e4b"  # gerador e revisor fixos: e4b -> 12b, sem WhisperX
    assert cfg.quality_gate is True and cfg.quality_vote is False  # porteiro de fala sim; votação não (docs/experimento-modo1-v2.md)
    assert qe.apply_mode("quality-2", Config()).quality_gate is False


def test_quality_2_is_a_fixed_configuration_and_is_listed(client_factory):
    cfg = qe.apply_mode("quality-2", Config())
    assert (cfg.quality_reviewer, cfg.quality_loops, cfg.quality_b_engine, cfg.quality_specialists) == ("gemma4:12b", 1, "whisperx", True)
    assert (cfg.quality_judge, cfg.quality_feedback, cfg.quality_self, cfg.quality_triage) == ("v2", "terms", False, False)
    assert qe.mode_generator("quality-2") == "gemma4:e4b" and qe.mode_generator("ollama") == "" and qe.mode_generator("quality") == ""
    base = Config()
    assert qe.apply_mode("quality", base) is base and qe.apply_mode("ollama", base) is base and base.quality_loops == 0
    c, _ = client_factory()
    engs = c.get("/api/engines").get_json()["engines"]
    assert [e["id"] for e in engs][:3] == ["ollama", "quality-1", "quality-2"]
    assert engs[1]["fixed_models"] == {"generator": "gemma4:e4b", "reviewer": "gemma4:12b"}
    assert engs[2]["fixed_models"] == {"generator": "gemma4:e4b", "reviewer": "gemma4:12b"}
    for mode in ("quality-1", "quality-2", "quality"):
        assert c.get(f"/api/models?engine={mode}").get_json()["models"][0]["name"] == "fake-audio:1b"


def test_quality_mode_runs_end_to_end(client_factory, monkeypatch, tmp_path, media_dir):
    sess = Session(["Texto do trecho número 1. Segunda frase do trecho."] * 50)

    class FakeWX(Engine):
        id, label, module, pip = "whisperx", "WhisperX", "os", "x"

        def catalog(self):
            return [{"name": "large-v3-turbo", "approx_size": "", "note": ""}]

        def is_installed(self, model):
            return True

        def session(self, model, *, device="auto", context="", job=None):
            return sess

    monkeypatch.setitem(engines.ENGINES, "whisperx", FakeWX())
    monkeypatch.delenv("MT_KEEP_WORK", raising=False)
    orig = pipeline._run_job

    def patched(job, client, config):
        config.work_dir = str(tmp_path / "work")
        return orig(job, client, config)

    monkeypatch.setattr(pipeline, "_run_job", patched)
    src = tmp_path / "m.mp3"
    shutil.copy(media_dir / "a.mp3", src)
    c, fake = client_factory()
    r = start(c, src, engine="quality-2")
    assert r.status_code == 202, r.get_json()
    s = wait_done(c, r.get_json()["id"])[0]
    assert s["status"] == "done", s
    rep = s["report"]
    assert rep["loops"] == 1 and rep["judge"] == "v2" and "WhisperX" in rep["b_source"] and "specialists" in rep and "texts" not in rep
    assert fake.pulled == ["gemma4:e4b", "gemma4:12b"]  # modelos fixos do modo: baixados se faltarem (o modelo da tela é ignorado)
    assert fake.calls and all(call[0] == "gemma4:e4b" for call in fake.calls)  # o gerador A é sempre o fixo
    assert sess.closed and not (tmp_path / "work" / r.get_json()["id"]).exists()


def test_quality_1_runs_end_to_end_without_whisperx(client_factory, monkeypatch, tmp_path, media_dir):
    monkeypatch.delenv("MT_KEEP_WORK", raising=False)
    orig = pipeline._run_job

    def patched(job, client, config):
        config.work_dir = str(tmp_path / "work")
        return orig(job, client, config)

    monkeypatch.setattr(pipeline, "_run_job", patched)
    src = tmp_path / "m1.mp3"
    shutil.copy(media_dir / "a.mp3", src)
    c, fake = client_factory()
    r = start(c, src, engine="quality-1")
    assert r.status_code == 202, r.get_json()
    s = wait_done(c, r.get_json()["id"])[0]
    assert s["status"] == "done", s
    rep = s["report"]
    assert rep["judge"] == "self" and rep["loops"] == 1 and rep["b_source"] == "" and rep["reviewed"] >= 1
    assert fake.pulled == ["gemma4:e4b", "gemma4:12b"] and all(call[0] == "gemma4:e4b" for call in fake.calls)  # modelos fixos, sem motor Whisper
