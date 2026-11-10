import json
import time

import pytest

import io
import os
import zipfile

from transcritor import ffmpeg_manager as fm
from transcritor import ollama_manager as om
from transcritor import setup as setup_mod
from transcritor.ollama_client import OllamaError
from transcritor.settings import Settings, SettingsError


class FakeOllama:
    def __init__(self, models=(), pull_adds=None, pull_error=None):
        self.models, self.pull_adds, self.pull_error = list(models), pull_adds, pull_error
        self.host = ""
        self.pulled = []

    def audio_models(self):
        return [{"name": m, "size": 1} for m in self.models]

    def pull(self, model, progress=None):
        self.pulled.append(model)
        if self.pull_error:
            raise OllamaError(self.pull_error)
        progress(5, 10, "pulling")
        progress(10, 10, "success")
        self.models.append(self.pull_adds or model)


class FakeProc:
    def __init__(self):
        self.started = None

    def start(self, binary):
        self.started = binary

    def stop(self):
        pass


def make(tmp_path, client, monkeypatch, *, running=True, binary="/bin/ollama", ffmpeg=True, **settings):
    st = Settings(str(tmp_path / "s.json"))
    if settings:
        st.update(settings, save=False)
    state = {"running": running}
    proc = FakeProc()
    monkeypatch.setattr(om, "is_running", lambda host, timeout=2: state["running"])
    monkeypatch.setattr(om, "find_binary", lambda: binary)
    monkeypatch.setattr(fm, "resolve", lambda configured="": ("/x/ffmpeg", "/x/ffprobe") if ffmpeg else None)
    monkeypatch.setattr(fm, "can_auto_install", lambda: False)  # nunca baixa de verdade nos testes
    monkeypatch.setattr(setup_mod.time, "sleep", lambda s: state.update(running=True))
    mgr = setup_mod.SetupManager(st, client, proc)
    return mgr, proc


def run(mgr):
    mgr.run()
    mgr._thread.join(10)
    return mgr.snapshot()


def test_ready_with_existing_audio_model_no_download(tmp_path, monkeypatch):
    c = FakeOllama(models=["qualquer:audio"])
    mgr, proc = make(tmp_path, c, monkeypatch)
    s = run(mgr)
    assert s["status"] == "ready" and c.pulled == [] and proc.started is None


def test_pulls_base_model_when_no_audio_model(tmp_path, monkeypatch):
    c = FakeOllama()
    mgr, _ = make(tmp_path, c, monkeypatch)
    s = run(mgr)
    assert c.pulled == ["gemma4:e4b"] and s["status"] == "ready"


def test_auto_pull_disabled(tmp_path, monkeypatch):
    c = FakeOllama()
    mgr, _ = make(tmp_path, c, monkeypatch, auto_pull=False)
    s = run(mgr)
    assert s["status"] == "error" and c.pulled == []


def test_pull_failure_and_non_audio_model(tmp_path, monkeypatch):
    mgr, _ = make(tmp_path, FakeOllama(pull_error="sem rede"), monkeypatch)
    assert "sem rede" in run(mgr)["message"]
    # modelo baixado não aparece entre os de áudio
    c = FakeOllama(pull_adds="outro:1b")
    mgr, _ = make(tmp_path, c, monkeypatch)
    s = run(mgr)
    assert s["status"] == "error" and "áudio" in s["message"]


def test_starts_local_server_when_stopped(tmp_path, monkeypatch):
    mgr, proc = make(tmp_path, FakeOllama(models=["m"]), monkeypatch, running=False)
    s = run(mgr)
    assert proc.started == "/bin/ollama" and s["status"] == "ready"


def test_ollama_missing(tmp_path, monkeypatch):
    mgr, proc = make(tmp_path, FakeOllama(), monkeypatch, running=False, binary=None)
    s = run(mgr)
    assert s["status"] == "ollama_missing" and "requisito" not in s["message"] and proc.started is None
    assert s["download_page"].startswith("https://ollama.com")


def test_remote_host_unreachable_is_error_not_start(tmp_path, monkeypatch):
    mgr, proc = make(tmp_path, FakeOllama(), monkeypatch, running=False, ollama_host="http://10.0.0.5:11434")
    monkeypatch.setattr(setup_mod.time, "sleep", lambda s: None)  # remoto: espera, mas nunca inicia local
    mgr.remote_wait_attempts = 2
    s = run(mgr)
    assert s["status"] == "error" and proc.started is None


def test_ffmpeg_missing_without_auto_install(tmp_path, monkeypatch):
    mgr, _ = make(tmp_path, FakeOllama(), monkeypatch, ffmpeg=False)
    s = run(mgr)
    assert s["status"] == "ffmpeg_missing" and "ffmpeg" in s["detail"].lower()


def test_ffmpeg_downloaded_locally_when_absent(tmp_path, monkeypatch):
    c = FakeOllama(models=["m"])
    mgr, _ = make(tmp_path, c, monkeypatch, ffmpeg=False)
    monkeypatch.setattr(fm, "can_auto_install", lambda: True)
    monkeypatch.setattr(fm, "download_local", lambda progress=None: (progress(5, 10), ("/loc/ffmpeg", "/loc/ffprobe"))[1])
    s = run(mgr)
    assert s["status"] == "ready" and fm.ffmpeg_path() == "/loc/ffmpeg"
    fm.set_active(None, None)


def test_ffmpeg_download_failure_reports_error(tmp_path, monkeypatch):
    mgr, _ = make(tmp_path, FakeOllama(models=["m"]), monkeypatch, ffmpeg=False)
    monkeypatch.setattr(fm, "can_auto_install", lambda: True)

    def boom(progress=None):
        raise RuntimeError("sem rede")

    monkeypatch.setattr(fm, "download_local", boom)
    s = run(mgr)
    assert s["status"] == "ffmpeg_missing" and "sem rede" in s["message"]


def test_configured_ffmpeg_path_invalid_does_not_download(tmp_path, monkeypatch):
    mgr, _ = make(tmp_path, FakeOllama(models=["m"]), monkeypatch, ffmpeg=False)
    mgr.settings._data["ffmpeg_path"] = "/nao/existe"
    monkeypatch.setattr(fm, "can_auto_install", lambda: True)
    monkeypatch.setattr(fm, "download_local", lambda progress=None: pytest.fail("não deveria baixar"))
    s = run(mgr)
    assert s["status"] == "ffmpeg_missing" and "configurado" in s["message"]


def _touch(d, *names):
    os.makedirs(d, exist_ok=True)
    for n in names:
        open(os.path.join(d, n), "w").close()


def test_resolve_order(tmp_path, monkeypatch):
    ext = ".exe" if os.name == "nt" else ""
    cfg, local = tmp_path / "cfg", tmp_path / "local"
    _touch(cfg / "bin", "ffmpeg" + ext, "ffprobe" + ext)
    _touch(local, "ffmpeg" + ext, "ffprobe" + ext)
    monkeypatch.setenv("MINITRANSCRITOR_FFMPEG_DIR", str(local))
    monkeypatch.setattr(fm.shutil, "which", lambda t: None)
    # 1) configurado (pasta raiz com bin/, pasta direta ou o próprio exe)
    assert fm.resolve(str(cfg))[0] == str(cfg / "bin" / ("ffmpeg" + ext))
    assert fm.resolve(str(cfg / "bin" / ("ffmpeg" + ext)))[1] == str(cfg / "bin" / ("ffprobe" + ext))
    # configurado inválido não cai silenciosamente em outro
    assert fm.resolve(str(tmp_path / "nada")) is None
    # 2) PATH tem prioridade sobre a pasta local
    monkeypatch.setattr(fm.shutil, "which", lambda t: f"/sys/{t}")
    assert fm.resolve("") == ("/sys/ffmpeg", "/sys/ffprobe")
    # 3) pasta local
    monkeypatch.setattr(fm.shutil, "which", lambda t: None)
    assert fm.resolve("")[0] == str(local / ("ffmpeg" + ext))
    # 4) nada
    monkeypatch.setenv("MINITRANSCRITOR_FFMPEG_DIR", str(tmp_path / "vazio"))
    assert fm.resolve("") is None


def test_download_local_extracts_only_tools(tmp_path, monkeypatch):
    ext = ".exe" if os.name == "nt" else ""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n in ("ffmpeg", "ffprobe", "ffplay"):
            z.writestr(f"ffmpeg-8.0-essentials_build/bin/{n}{ext}", b"bin")
        z.writestr("ffmpeg-8.0-essentials_build/LICENSE", "x")

    class Resp:
        headers = {"content-length": str(len(buf.getvalue()))}
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def raise_for_status(self): pass
        def iter_content(self, n): yield buf.getvalue()

    monkeypatch.setenv("MINITRANSCRITOR_FFMPEG_DIR", str(tmp_path / "ff"))
    monkeypatch.setattr(fm.requests, "get", lambda *a, **k: Resp())
    monkeypatch.setattr(fm, "verify", lambda p: True)
    seen = []
    ff, fp = fm.download_local(lambda d, t: seen.append((d, t)))
    assert os.path.isfile(ff) and os.path.isfile(fp) and seen
    assert sorted(os.listdir(tmp_path / "ff")) == sorted(["ffmpeg" + ext, "ffprobe" + ext])


def test_settings_rejects_invalid_ffmpeg_path(tmp_path):
    s = Settings(str(tmp_path / "s.json"))
    with pytest.raises(SettingsError):
        s.update({"ffmpeg_path": str(tmp_path)})
    assert s.update({"ffmpeg_path": ""})["ffmpeg_path"] == ""


def test_settings_validation_and_persistence(tmp_path):
    p = tmp_path / "s.json"
    s = Settings(str(p))
    assert s["base_model"] == "gemma4:e4b" and s["auto_pull"] is True
    s.update({"ollama_host": "localhost:11434", "chunk_seconds": 10, "auto_pull": False})
    again = Settings(str(p))
    assert again["ollama_host"] == "http://localhost:11434" and again["chunk_seconds"] == 10 and again["auto_pull"] is False
    for bad in ({"chunk_seconds": 1}, {"chunk_seconds": "x"}, {"base_model": " "}, {"base_model": "a b"}):
        with pytest.raises(SettingsError):
            s.update(bad)
    p.write_text("{corrompido")
    assert Settings(str(p))["base_model"] == "gemma4:e4b"


def test_settings_api(client_factory):
    c, _ = client_factory()
    assert c.get("/api/settings").get_json()["base_model"] == "gemma4:e4b"
    assert c.get("/api/setup").get_json()["status"] == "ready"
    assert c.post("/api/settings", json={"chunk_seconds": 99}).status_code == 400
    assert c.post("/api/settings", json={"chunk_seconds": 8, "base_model": "x:1"}).get_json()["chunk_seconds"] == 8


def test_is_local_host():
    assert om.is_local_host("http://127.0.0.1:11434") and om.is_local_host("http://localhost:1")
    assert not om.is_local_host("http://192.168.0.2:11434")
