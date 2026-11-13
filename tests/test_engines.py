import importlib.machinery
import hashlib
import shutil
import sys
import types

import pytest

from transcritor import engines
from transcritor.engines import Engine, EngineError, Seg

from .test_api import start, wait_done


class FakeEngine(Engine):
    id = "fake-eng"
    label = "Fake"
    description = "motor de teste"
    module = "os"  # sempre importável => "instalado"
    pip = "pip install fake"
    supports_diarization = True

    def __init__(self, lang="en", diarization=True):
        self.installed = set()
        self.downloaded, self.calls = [], []
        self.lang = lang
        self.supports_diarization = diarization

    def catalog(self):
        return [{"name": "m-a", "approx_size": "~1 GB", "note": "a"}, {"name": "m-b", "approx_size": "~2 GB", "note": "b"}]

    def is_installed(self, model):
        return model in self.installed

    def ensure_model(self, model, progress=None):
        progress(5, 10)
        progress(10, 10)
        self.installed.add(model)
        self.downloaded.append(model)

    def transcribe(self, model, samples, chunks, job, upd, *, diarize=False, hf_token="", parallel=1, context="", device="auto"):
        upd(50, "transcrevendo")
        self.calls.append((model, diarize, hf_token))
        self.context = context
        return [Seg(1, 3, "Hello everyone.", "SPEAKER_01"), Seg(3.5, 6, "How are you?", "SPEAKER_00"),
                Seg(6.5, 8, "Fine.", "SPEAKER_00")], self.lang


@pytest.fixture
def fake_engine(monkeypatch):
    eng = FakeEngine()
    monkeypatch.setitem(engines.ENGINES, eng.id, eng)
    return eng


@pytest.fixture
def audio_file(media_dir, tmp_path):
    src = tmp_path / "entrada.mp3"
    shutil.copy(media_dir / "a.mp3", src)
    return src


# ---------- catálogo ----------

def test_bundled_catalog_is_valid_and_has_no_ambiguous_names():
    cat = engines.load_catalog()
    assert set(cat) == {"openai-whisper", "faster-whisper", "whisperx", "whisper-cpp", "nemo-speech-cpp", "nemotron-vllm"}
    for eid, models in cat.items():
        names = [m["name"] for m in models]
        assert all(m["approx_size"] and m["note"] for m in models) and len(names) == len(set(names))
        if eid in ("nemo-speech-cpp", "nemotron-vllm"):  # família Nemotron: nomes próprios (ver test_nemocpp.py)
            continue
        assert "large-v3" in names and any(n.startswith("large-v3-turbo") for n in names)
        assert not {"large", "turbo", "latest"} & set(names)  # sempre o nome específico
        assert names[0].startswith("large-v3-turbo")  # recomendado primeiro
        assert all(m["approx_size"] and m["note"] for m in models)


def test_parse_catalog_ignores_bad_lines():
    cat = engines.parse_catalog("# c\n\nfaster-whisper | small | ~0.5 GB | ok\nfaster-whisper | small | dup\nx | com espaco | 1\nsolta\n")
    assert cat == {"faster-whisper": [{"name": "small", "approx_size": "~0.5 GB", "note": "ok"}]}


# ---------- disponibilidade ----------

def test_missing_package_reports_install_command(monkeypatch):
    monkeypatch.setattr(engines.importlib.util, "find_spec", lambda name: None)
    for eid, cmd in (("openai-whisper", "openai-whisper"), ("faster-whisper", "faster-whisper"), ("whisperx", "whisperx")):
        st = engines.get(eid).status()
        assert not st["available"] and f"pip install {cmd}" in st["hint"]
        assert engines.get(eid).models() == [] or all(not m["installed"] for m in engines.get(eid).models())


def test_only_whisperx_supports_diarization():
    assert [e.id for e in engines.ENGINES.values() if e.supports_diarization] == ["whisperx"]


def test_engines_endpoint_lists_ollama_first(client_factory, fake_engine):
    c, _ = client_factory()
    body = c.get("/api/engines").get_json()
    assert body["engines"][0]["id"] == "ollama" and body["engines"][0]["available"]
    assert {e["id"] for e in body["engines"]} >= {"ollama", "openai-whisper", "faster-whisper", "whisperx", "fake-eng"}


def test_models_endpoint_marks_only_downloaded_as_installed(client_factory, fake_engine):
    fake_engine.installed.add("m-b")
    c, _ = client_factory()
    models = c.get("/api/models?engine=fake-eng").get_json()["models"]
    assert [(m["name"], m["installed"]) for m in models] == [("m-b", True), ("m-a", False)]  # baixados primeiro
    assert c.get("/api/models?engine=nao-existe").get_json()["ok"] is False


def test_models_endpoint_for_uninstalled_engine_explains_how_to_install(client_factory, monkeypatch):
    monkeypatch.setattr(engines.importlib.util, "find_spec", lambda name: None)
    c, _ = client_factory()
    body = c.get("/api/models?engine=whisperx").get_json()
    assert body["ok"] is False and "pip install whisperx" in body["error"]


# ---------- validação do pedido ----------

def test_job_validation(client_factory, audio_file, fake_engine):
    c, _ = client_factory()
    assert start(c, audio_file, engine="nao-existe", model="m-a").status_code == 400
    assert start(c, audio_file, engine="fake-eng", model="nao-tem").status_code == 400
    r = start(c, audio_file, engine="fake-eng", model="m-a", diarize=True)  # sem token do Hugging Face
    assert r.status_code == 400 and "Hugging Face" in r.get_json()["error"]
    assert start(c, audio_file, diarize=True).status_code == 400  # Ollama não faz diarização
    fake_engine.supports_diarization = False
    assert "não faz diarização" in start(c, audio_file, engine="fake-eng", model="m-a", diarize=True).get_json()["error"]


# ---------- execução ----------

def read_srt(s):
    with open(s["output_path"], encoding="utf-8") as f:
        return f.read()


def test_job_downloads_model_translates_and_uses_real_timestamps(client_factory, audio_file, fake_engine):
    c, fake = client_factory()
    r = start(c, audio_file, engine="fake-eng", model="m-a")
    assert r.status_code == 202
    s, seen = wait_done(c, r.get_json()["id"])
    assert s["status"] == "done", s
    assert fake_engine.downloaded == ["m-a"] and fake_engine.calls == [("m-a", False, "")]
    assert seen == sorted(seen) and s["progress"] == 100 and s["engine"] == "fake-eng"
    assert fake.batches == [3]  # detectado 'en' e saída pt-BR: traduz com o modelo do Ollama, em lote
    srt = read_srt(s)
    assert "00:00:01,000 --> 00:00:03,000" in srt  # timestamps do motor, não proporcionais
    assert "[Falante" not in srt


def test_already_downloaded_model_is_not_downloaded_again(client_factory, audio_file, fake_engine):
    fake_engine.installed.add("m-a")
    c, _ = client_factory()
    s, _ = wait_done(c, start(c, audio_file, engine="fake-eng", model="m-a").get_json()["id"])
    assert s["status"] == "done" and fake_engine.downloaded == []


def test_same_language_skips_translation(client_factory, audio_file, monkeypatch):
    eng = FakeEngine(lang="pt")
    monkeypatch.setitem(engines.ENGINES, eng.id, eng)
    eng.installed.add("m-a")
    c, fake = client_factory()
    s, _ = wait_done(c, start(c, audio_file, engine="fake-eng", model="m-a").get_json()["id"])
    assert s["status"] == "done" and not getattr(fake, "batches", []) and not getattr(fake, "translations", [])


def test_diarization_labels_speakers_in_order_of_appearance(client_factory, audio_file, fake_engine):
    fake_engine.installed.add("m-a")
    c, _ = client_factory()
    c.post("/api/settings", json={"hf_token": "hf_segredo"})
    s, _ = wait_done(c, start(c, audio_file, engine="fake-eng", model="m-a", diarize=True).get_json()["id"])
    assert s["status"] == "done", s
    assert fake_engine.calls[-1] == ("m-a", True, "hf_segredo")
    srt = read_srt(s)
    assert srt.count("[Falante 1]") == 1 and srt.count("[Falante 2]") == 1  # só quando o falante muda
    assert srt.index("[Falante 1]") < srt.index("[Falante 2]")


def test_no_ollama_model_for_translation_keeps_original_with_warning(client_factory, audio_file, fake_engine):
    fake_engine.installed.add("m-a")
    c, fake = client_factory()
    fake.audio_models = lambda: []
    s, _ = wait_done(c, start(c, audio_file, engine="fake-eng", model="m-a").get_json()["id"])
    assert s["status"] == "done" and any("traduzir" in w for w in s["warnings"])


def test_server_mode_never_downloads_models():
    from transcritor.jobs import Job
    from transcritor.pipeline import _ensure_engine_model

    eng = FakeEngine()
    job = Job(id="x", path="p", language="pt-BR", model="m-a", engine="fake-eng", owner="a@b.c")
    with pytest.raises(EngineError):
        _ensure_engine_model(job, eng)
    assert eng.downloaded == []


# ---------- token do Hugging Face ----------

def test_hf_token_is_never_sent_back_to_the_browser(client_factory):
    c, _ = client_factory()
    assert c.post("/api/settings", json={"hf_token": "hf_abc"}).get_json()["ok"]
    body = c.get("/api/settings").get_json()
    assert body["hf_token_set"] is True and "hf_token" not in body
    assert "hf_token" not in c.post("/api/settings", json={"chunk_seconds": 12}).get_json()
    assert c.get("/api/settings").get_json()["hf_token_set"] is True  # não enviar = manter
    assert c.post("/api/settings", json={"hf_token": "com espaco"}).status_code == 400
    c.post("/api/settings", json={"hf_token": ""})  # limpa (e não deixa vazar em outros testes)
    assert c.get("/api/settings").get_json()["hf_token_set"] is False


# ---------- download do OpenAI Whisper (HTTP simulado) ----------

class FakeResponse:
    def __init__(self, data):
        self.data = data
        self.headers = {"content-length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, n):
        for i in range(0, len(self.data), n):
            yield self.data[i : i + n]


@pytest.fixture
def fake_whisper_pkg(monkeypatch, tmp_path):
    mod = types.ModuleType("whisper")
    mod.__spec__ = importlib.machinery.ModuleSpec("whisper", None)
    data = b"pesos-do-modelo" * 1000
    sha = hashlib.sha256(data).hexdigest()
    mod._MODELS = {"tiny": f"https://exemplo.test/models/{sha}/tiny.pt", "base": "https://exemplo.test/models/" + "0" * 64 + "/base.pt"}
    monkeypatch.setitem(sys.modules, "whisper", mod)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    import requests

    monkeypatch.setattr(requests, "get", lambda url, **kw: FakeResponse(data))
    return data


def test_openai_whisper_download_verifies_sha256(fake_whisper_pkg):
    eng = engines.get("openai-whisper")
    assert not eng.is_installed("tiny")
    seen = []
    eng.ensure_model("tiny", lambda d, t: seen.append((d, t)))
    assert eng.is_installed("tiny") and seen[-1][0] == seen[-1][1] > 0


def test_openai_whisper_corrupted_download_is_rejected(fake_whisper_pkg):
    eng = engines.get("openai-whisper")
    with pytest.raises(EngineError, match="corrompido"):
        eng.ensure_model("base")  # SHA esperado não confere
    assert not eng.is_installed("base")  # nada fica no cache
    with pytest.raises(EngineError, match="não conhece"):
        eng.ensure_model("large-v3")  # pacote não lista o modelo


def test_catalog_has_no_previous_generation_or_english_only_models():
    for eid, models in engines.load_catalog().items():
        for m in models:
            n = m["name"]
            assert not n.startswith("distil-") and ".en" not in n and not n.startswith(("large-v1", "large-v2")), (eid, n)
