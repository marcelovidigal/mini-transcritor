import os
import shutil

import numpy as np
import pytest

from transcritor import engines, pipeline, workdir
from transcritor import quality_engine as qe
from transcritor.audio import Chunk
from transcritor.config import Config
from transcritor.engines import Engine, Seg
from transcritor.jobs import Cancelled
from transcritor.ollama_client import OllamaError

from .test_api import start, wait_done


# ---------- portão e juiz (código puro) ----------

def test_gate_flags_divergent_empty_and_one_sided_fragments():
    a = ["olá mundo como vai", "tudo bem por aqui hoje", "", "só em A", "igual aqui"]
    b = ["olá mundo como vai", "tudo mal por lá hoje", "", "", "igual aqui"]
    assert qe.gate(a, b, "pt-BR") == [1, 3]  # divergem / só um tem texto; vazios nos dois e iguais passam
    assert qe.gate(["25 minutos"], ["vinte e cinco minutos"], "pt-BR") == []  # números por extenso não divergem


def test_judge_keeps_A_when_in_doubt_and_applies_guardrails():
    a, b = "Segundo o doutor maw, alguns pacientes", "Segundo o doutor Moll, alguns pacientes"
    assert qe.judge(a, b, None, "pt") == (a, "A")
    assert qe.judge(a, b, {"verdict": "B", "text": b, "confidence": 0.9}, "pt") == (b, "B")
    assert qe.judge(a, b, {"verdict": "A", "text": a, "confidence": 0.9}, "pt") == (a, "A")
    assert qe.judge(a, b, {"verdict": "B", "text": b, "confidence": 0.2}, "pt")[1] == "A(confiança baixa)"
    assert qe.judge(a, b, {"verdict": "B", "text": b, "confidence": "x"}, "pt")[1] == "A(confiança baixa)"
    assert qe.judge(a, "", {"verdict": "B", "text": "", "confidence": 0.9}, "pt")[1] == "A(B vazio)"
    assert qe.judge(a, b, {"verdict": "???", "text": "", "confidence": 0.9}, "pt")[1] == "A(veredito inválido)"
    # mescla: só com palavras de A, B ou do glossário, e perto de uma das duas
    ok = "Segundo o doutor Moll, alguns pacientes"
    assert qe.judge(a, b, {"verdict": "merge", "text": ok, "confidence": 0.9}, "pt") == (ok, "merge")
    inventado = "Segundo o doutor Zeferino, alguns pacientes"
    assert qe.judge(a, b, {"verdict": "merge", "text": inventado, "confidence": 0.9}, "pt")[1] == "A(mescla rejeitada)"
    assert qe.judge(a, b, {"verdict": "merge", "text": inventado, "confidence": 0.9}, "pt", glossary="Zeferino")[1] == "merge"
    longe = "texto completamente diferente que reescreve tudo o que foi falado"
    assert qe.judge(a, b, {"verdict": "merge", "text": longe, "confidence": 0.9}, "pt")[1] == "A(mescla rejeitada)"


def test_group_segments_by_fragment_midpoint():
    chunks = [Chunk(0, 10, np.zeros(1)), Chunk(10, 20, np.zeros(1)), Chunk(20, 30, np.zeros(1))]
    segs = [Seg(1, 4, "um"), Seg(8, 12, "dois"), Seg(14, 18, "três"), Seg(27, 29, "quatro")]  # "dois": ponto médio 10 → fragmento 1
    assert qe.group_by_fragment(segs, chunks) == ["um", "dois três", "quatro"]


# ---------- pasta de trabalho ----------

def test_workdir_lifecycle(tmp_path, monkeypatch):
    cfg = Config(work_dir=str(tmp_path))
    monkeypatch.delenv("MT_KEEP_WORK", raising=False)
    d = workdir.create(cfg, "job1")
    workdir.save_fragments(d, [Chunk(0, 1, np.zeros(16000, dtype=np.int16)), Chunk(1, 2, np.zeros(16000, dtype=np.int16))])
    assert sorted(os.listdir(d)) == ["0000.wav", "0001.wav"]
    monkeypatch.setenv("MT_KEEP_WORK", "1")
    workdir.remove(d)
    assert os.path.isdir(d)  # MT_KEEP_WORK mantém para análise
    monkeypatch.delenv("MT_KEEP_WORK")
    workdir.remove(d)
    assert not os.path.exists(d)
    old = workdir.create(cfg, "velha")
    os.utime(old, (1, 1))
    new = workdir.create(cfg, "nova")
    assert workdir.purge_old(cfg, days=7) == 1 and not os.path.exists(old) and os.path.isdir(new)


# ---------- pipeline completo com dublês ----------

class FakeWhisper(Engine):
    id = "fake-whisper"
    label = "FakeWhisper"
    module = "os"
    pip = "x"

    def catalog(self):
        return [{"name": "m", "approx_size": "", "note": ""}]

    def is_installed(self, model):
        return True

    def transcribe(self, model, samples, chunks, job, upd, *, diarize=False, hf_token="", parallel=1, context="", device="auto"):
        # um segmento por fragmento; o do fragmento 1 discorda do Ollama
        return [Seg(c.start + 0.1, c.end - 0.1,
                    "Outro texto completamente diferente aqui." if i == 1 else f"Texto do trecho número {i + 1}. Segunda frase do trecho.")
                for i, c in enumerate(chunks)], "pt"


@pytest.fixture
def quality_setup(client_factory, monkeypatch, tmp_path, media_dir):
    eng = FakeWhisper()
    monkeypatch.setitem(engines.ENGINES, eng.id, eng)
    monkeypatch.delenv("MT_KEEP_WORK", raising=False)
    work = tmp_path / "work"
    orig = pipeline._run_job
    jobs = []

    def patched(job, client, config):
        config.quality_b_engine, config.quality_b_model, config.work_dir = "fake-whisper", "m", str(work)
        jobs.append(job)
        return orig(job, client, config)

    monkeypatch.setattr(pipeline, "_run_job", patched)
    src = tmp_path / "q.mp3"
    shutil.copy(media_dir / "a.mp3", src)
    c, fake = client_factory()
    return c, fake, src, work, jobs


def run_quality(c, src, **over):
    r = start(c, src, engine="quality", **over)
    assert r.status_code == 202, r.get_json()
    return wait_done(c, r.get_json()["id"])[0]


def test_quality_engine_end_to_end_with_fakes(quality_setup):
    c, fake, src, work, jobs = quality_setup
    s = run_quality(c, src)
    assert s["status"] == "done", s
    rep = s["report"]
    assert rep["fragments"] >= 2 and rep["flagged"] >= 1 and rep["reviewed"] == rep["flagged"]
    assert rep["from_b"] == rep["flagged"]  # o FakeClient.review devolve veredito B com confiança 0,9
    assert fake.reviews and fake.reviews[0][0] == "fake-audio:1b"
    assert not (work / s["id"]).exists()  # concluído: fragmentos apagados


def test_quality_engine_removes_work_dir_on_error_and_keeps_on_cancel(quality_setup, monkeypatch):
    c, fake, src, work, jobs = quality_setup

    def boom(self, *a, **k):
        raise engines.EngineError("falha da 2ª geração")

    monkeypatch.setattr(FakeWhisper, "transcribe", boom)
    s = run_quality(c, src)
    assert s["status"] == "error" and "2ª geração" in s["error"]
    assert not (work / jobs[0].id).exists()  # erro: apaga

    def cancel(self, model, samples, chunks, job, upd, **k):
        raise Cancelled()

    monkeypatch.setattr(FakeWhisper, "transcribe", cancel)
    s = run_quality(c, src)
    assert s["status"] == "cancelled"
    assert (work / jobs[1].id).is_dir()  # cancelado: os fragmentos ficam (regra literal do usuário)


def test_quality_falls_back_to_a_second_ollama_model_when_whisper_is_missing(client_factory, media_dir, tmp_path, monkeypatch):
    monkeypatch.delenv("MT_KEEP_WORK", raising=False)
    monkeypatch.setattr(workdir, "base_dir", lambda config: str(tmp_path / "work"))  # nunca toca na pasta work/ real
    src = tmp_path / "f.mp3"
    shutil.copy(media_dir / "a.mp3", src)
    orig = pipeline._run_job

    def patched(job, client, config):
        config.quality_b_engine, config.work_dir = "motor-inexistente", str(tmp_path / "work")
        return orig(job, client, config)

    monkeypatch.setattr(pipeline, "_run_job", patched)
    c, fake = client_factory()
    s = run_quality(c, src)  # motor B ausente
    assert s["status"] == "done", s
    assert any("indisponível para a 2ª geração" in w for w in s["warnings"]) and "Ollama" in str(s["report"]["b_source"])


def test_quality_in_engine_list_and_validation(client_factory, media_dir, tmp_path):
    c, _ = client_factory()
    ids = [e["id"] for e in c.get("/api/engines").get_json()["engines"]]
    assert ids[:3] == ["ollama", "quality-1", "quality-2"] and "quality" not in ids
    assert c.get("/api/models?engine=quality").get_json()["models"][0]["name"] == "fake-audio:1b"
    src = tmp_path / "x.mp3"
    shutil.copy(media_dir / "a.mp3", src)
    assert start(c, src, engine="quality", diarize=True).status_code == 400  # sem diarização


def test_judge_prefer_B_changes_only_the_fallback():
    a, b = "texto da geração A aqui", "texto da geração B aqui"
    assert qe.judge(a, b, None, "pt", prefer="B") == (b, "B")
    assert qe.judge(a, b, {"verdict": "B", "text": b, "confidence": 0.1}, "pt", prefer="B")[1] == "B(confiança baixa)"
    assert qe.judge(a, b, {"verdict": "A", "text": a, "confidence": 0.9}, "pt", prefer="B") == (a, "A")  # o revisor pode escolher A
    assert qe.judge(a, "", None, "pt", prefer="B") == (a, "A")  # B vazio: cai para A
    assert qe.judge(a, b, {"verdict": "???", "text": "", "confidence": 0.9}, "pt", prefer="B")[1] == "B(veredito inválido)"


# ---------- três gerações (A, B, C) ----------

def test_gate_multi_flags_any_divergent_pair_or_missing_text():
    a = ["igual aqui", "tudo bem por aqui hoje", "só A tem", "x y z w"]
    b = ["igual aqui", "tudo bem por aqui hoje", "só A tem", "x y z w"]
    c = ["igual aqui", "tudo mal por lá hoje", "", "x y z w"]
    assert qe.gate_multi([a, b, c], "pt") == [1, 2]  # C diverge no 1; C vazio no 2
    assert qe.gate_multi([a, b], "pt") == []  # com só duas, concordam
    assert qe.gate(a, c, "pt") == [1, 2]


def test_central_picks_the_generation_closest_to_the_others():
    a, b, c = "o gato subiu no telhado ontem", "o gato subiu no telhado ontem", "um pato desceu do telhado hoje"
    assert qe.central([a, b, c], "pt") in (0, 1)
    assert qe.central([c, a, b], "pt") in (1, 2)  # a maioria (as duas iguais) vence
    assert qe.central(["", "texto único aqui", ""], "pt") == 1


def test_judge_with_three_generations_falls_back_to_the_majority_and_accepts_C():
    a, b, c = "o gato subiu no telhado ontem", "o gato subiu no telhado hoje", "o gato subiu no telhado hoje"
    text, src = qe.judge(a, b, None, "pt", c=c)
    assert text in (b, c) and src == "central"  # B e C concordam: a maioria vence A
    assert qe.judge(a, b, {"verdict": "C", "text": c, "confidence": 0.9}, "pt", c=c) == (c, "C")
    assert qe.judge(a, b, {"verdict": "A", "text": a, "confidence": 0.9}, "pt", c=c) == (a, "A")
    assert qe.judge(a, b, {"verdict": "C", "text": c, "confidence": 0.1}, "pt", c=c)[1] == "central(confiança baixa)"
    assert qe.judge(a, b, {"verdict": "C", "text": "", "confidence": 0.9}, "pt", c="")[1] == "A(C vazio)"  # sem C, veredito C não vale


def test_review_prompt_and_schema_include_C_only_when_given():
    from transcritor.ollama_client import build_review_prompt

    assert "C:" not in build_review_prompt("a", "b") and "Two" in build_review_prompt("a", "b")
    p = build_review_prompt("a", "b", c="c texto")
    assert "Three" in p and "C: c texto" in p and "A, B or C" in p


class ThreeGen(FakeWhisper):
    id = "fake-whisper-c"
    label = "FakeWhisperC"


def test_quality_engine_runs_with_three_generations(client_factory, monkeypatch, tmp_path, media_dir):
    eng = ThreeGen()
    monkeypatch.setitem(engines.ENGINES, eng.id, eng)
    monkeypatch.delenv("MT_KEEP_WORK", raising=False)
    work = tmp_path / "work"
    orig = pipeline._run_job

    def patched(job, client, config):
        config.quality_b_engine, config.quality_b_model = "ollama", "fake-audio:1b"  # B = outro modelo do Ollama
        config.quality_c_engine, config.quality_c_model, config.work_dir = "fake-whisper-c", "m", str(work)
        return orig(job, client, config)

    monkeypatch.setattr(pipeline, "_run_job", patched)
    src = tmp_path / "q3.mp3"
    shutil.copy(media_dir / "a.mp3", src)
    c, fake = client_factory()
    s = run_quality(c, src)
    assert s["status"] == "done", s
    rep = s["report"]
    assert len(rep["sources"]) == 3 and rep["sources"][0].startswith("Ollama") and rep["sources"][2].startswith("FakeWhisperC")
    assert rep["flagged"] >= 1 and rep["reviewed"] == rep["flagged"]
    assert {"geracao_a", "geracao_b", "geracao_c", "revisao"} <= set(rep["seconds"])


def test_unload_all_and_release_memory(monkeypatch):
    import sys
    import types

    from transcritor.ollama_client import OllamaClient

    sent = []
    c = OllamaClient("http://x")

    def fake(method, path, **kw):
        sent.append((method, path, kw.get("json")))
        return {"models": [{"name": "gemma4:e4b"}, {"name": "gemma4:12b"}]} if path == "/api/ps" else {}

    monkeypatch.setattr(c, "_request", fake)
    assert c.unload_all() == 2
    assert [s[2]["model"] for s in sent if s[1] == "/api/generate"] == ["gemma4:e4b", "gemma4:12b"]
    assert all(s[2]["keep_alive"] == 0 for s in sent if s[1] == "/api/generate")
    monkeypatch.setattr(c, "_request", lambda *a, **k: (_ for _ in ()).throw(OllamaError("fora do ar")))
    assert c.unload_all() == 0  # nunca levanta erro

    calls = []
    torch = types.ModuleType("torch")
    torch.xpu = types.SimpleNamespace(empty_cache=lambda: calls.append("xpu"))
    torch.cuda = types.SimpleNamespace(empty_cache=lambda: calls.append("cuda"))
    monkeypatch.setitem(sys.modules, "torch", torch)
    engines.release_memory()
    assert sorted(calls) == ["cuda", "xpu"]
