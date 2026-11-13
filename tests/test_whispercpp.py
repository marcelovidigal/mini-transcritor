import hashlib
import json
import os

import numpy as np
import pytest

from transcritor import engines
from transcritor.audio import Chunk
from transcritor.engines import EngineError, WhisperCppEngine
from transcritor.jobs import Cancelled


@pytest.fixture
def wcpp(monkeypatch, tmp_path):
    """Motor com executável e modelos falsos em pastas temporárias."""
    monkeypatch.setenv("MINITRANSCRITOR_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("WHISPER_CPP_PATH", raising=False)
    monkeypatch.delenv("MT_DEVICE", raising=False)
    from transcritor import whispercpp_manager as wm

    monkeypatch.setattr(wm, "has_nvidia", lambda: False)  # backend CPU, independente da máquina
    exe = tmp_path / "bin" / WhisperCppEngine._names()[0]
    exe.parent.mkdir()
    exe.write_text("")
    eng = engines.get("whisper-cpp")
    monkeypatch.setattr(eng, "configured_path", str(tmp_path / "bin"))
    return eng


def put_model(eng, name="tiny"):
    os.makedirs(eng.models_dir(), exist_ok=True)
    with open(eng._file(name), "wb") as f:
        f.write(b"0" * 1_100_000)


class StubJob:
    def __init__(self, cancel_after=None):
        self.warnings, self.n, self.cancel_after = [], 0, cancel_after

    def check_cancel(self):
        self.n += 1
        if self.cancel_after is not None and self.n > self.cancel_after:
            raise Cancelled()


class FakePopen:
    last = None
    stderr_lines = ["whisper_init: ...\n", "progress =  10%\n", "progress =  60%\n", "progress = 100%\n"]
    returncode_value = 0
    payload = {"result": {"language": "en"},
               "transcription": [{"offsets": {"from": 1000, "to": 3500}, "text": " Hello there."},
                                 {"offsets": {"from": 4000, "to": 6000}, "text": "  "},
                                 {"offsets": {"from": 6500, "to": 9000}, "text": " How are you?"}]}

    def __init__(self, cmd, **kw):
        FakePopen.last = self
        self.cmd, self.killed = cmd, False
        self.stderr = iter(FakePopen.stderr_lines)
        self.returncode = FakePopen.returncode_value
        if self.returncode == 0:
            with open(cmd[cmd.index("-of") + 1] + ".json", "w", encoding="utf-8") as f:
                json.dump(FakePopen.payload, f)

    def wait(self):
        return self.returncode

    def kill(self):
        self.killed = True


@pytest.fixture
def popen(monkeypatch):
    FakePopen.stderr_lines = ["whisper_init: ...\n", "progress =  10%\n", "progress =  60%\n", "progress = 100%\n"]
    FakePopen.returncode_value = 0
    monkeypatch.setattr(engines.subprocess, "Popen", FakePopen)
    return FakePopen


def samples():
    return np.zeros(16000, dtype=np.int16)


# ---------- localização do executável ----------

def test_binary_lookup_order_and_subfolders(monkeypatch, tmp_path):
    eng = WhisperCppEngine()
    monkeypatch.delenv("WHISPER_CPP_PATH", raising=False)
    from transcritor import whispercpp_manager as wm

    monkeypatch.setattr(wm, "app_dir", lambda: str(tmp_path / "app"))
    monkeypatch.setattr(engines.shutil, "which", lambda n: None)
    monkeypatch.setattr(eng, "configured_path", "")
    assert eng.find_binary() is None and not eng.installed()
    assert eng.available() == wm.supported("cpu")  # sem executável, mas dá para baixar (Windows/Linux)
    assert "whisper.cpp" in eng.not_installed_message()
    rel = tmp_path / "wc" / "Release"
    rel.mkdir(parents=True)
    (rel / WhisperCppEngine._names()[0]).write_text("")
    eng.configured_path = str(tmp_path / "wc")  # a pasta aponta para o pai: acha em Release/
    assert eng.find_binary().endswith(WhisperCppEngine._names()[0]) and eng.available()
    monkeypatch.setattr(eng, "configured_path", "")
    monkeypatch.setenv("WHISPER_CPP_PATH", str(rel))
    assert eng.find_binary() is not None


def test_settings_rejects_a_folder_without_whisper_cli(tmp_path, client_factory):
    c, _ = client_factory()
    assert c.post("/api/settings", json={"whispercpp_path": str(tmp_path)}).status_code == 400
    (tmp_path / WhisperCppEngine._names()[0]).write_text("")
    r = c.post("/api/settings", json={"whispercpp_path": str(tmp_path)})
    assert r.get_json()["ok"] and engines.get("whisper-cpp").configured_path == str(tmp_path)
    c.post("/api/settings", json={"whispercpp_path": ""})
    assert engines.get("whisper-cpp").configured_path == ""


def test_missing_binary_is_reported_in_status_and_models(client_factory, monkeypatch, tmp_path):
    from transcritor import whispercpp_manager as wm

    monkeypatch.setattr(wm, "app_dir", lambda: str(tmp_path / "app"))
    monkeypatch.setattr(engines.shutil, "which", lambda n: None)
    monkeypatch.setattr(engines.get("whisper-cpp"), "configured_path", "")
    monkeypatch.delenv("WHISPER_CPP_PATH", raising=False)
    c, _ = client_factory()
    st = next(e for e in c.get("/api/engines").get_json()["engines"] if e["id"] == "whisper-cpp")
    assert st["installed"] is False and st["needs_download"] == wm.supported("cpu")  # será baixado ao gerar
    monkeypatch.setattr(wm, "supported", lambda variant: False)  # plataforma sem binário pronto
    st = next(e for e in c.get("/api/engines").get_json()["engines"] if e["id"] == "whisper-cpp")
    assert not st["available"] and "whisper.cpp" in st["hint"]
    assert "whisper.cpp" in c.get("/api/models?engine=whisper-cpp").get_json()["error"]


# ---------- execução ----------

def test_transcribe_builds_command_and_parses_json(wcpp, popen):
    put_model(wcpp)
    seen = []
    segs, lang = wcpp.transcribe("tiny", samples(), [Chunk(0, 1, samples())], StubJob(),
                                 lambda p, stage=None: seen.append(p), context="Dra. Helena Duarte")
    cmd = popen.last.cmd
    assert cmd[0].endswith(WhisperCppEngine._names()[0]) and cmd[cmd.index("-m") + 1].endswith("ggml-tiny.bin")
    assert "-oj" in cmd and "-pp" in cmd and cmd[cmd.index("-l") + 1] == "auto" and "-ng" in cmd  # variante CPU: sempre sem GPU
    assert cmd[cmd.index("--prompt") + 1] == "Dra. Helena Duarte"
    assert [(s.start, s.end, s.text) for s in segs] == [(1.0, 3.5, "Hello there."), (6.5, 9.0, "How are you?")]  # texto vazio descartado
    assert lang == "en" and seen == [0, 10, 60, 99]


def test_cpu_device_disables_gpu(wcpp, popen):
    put_model(wcpp)
    wcpp.transcribe("tiny", samples(), [], StubJob(), lambda *a, **k: None, device="cpu")
    assert "-ng" in popen.last.cmd


def test_env_cpu_also_disables_gpu_when_setting_is_auto(wcpp, popen, monkeypatch):
    put_model(wcpp)
    monkeypatch.setenv("MT_DEVICE", "cpu")
    wcpp.transcribe("tiny", samples(), [], StubJob(), lambda *a, **k: None, device="auto")
    assert "-ng" in popen.last.cmd


def test_nonzero_exit_reports_stderr_tail(wcpp, popen):
    put_model(wcpp)
    popen.returncode_value = 3
    popen.stderr_lines = ["linha 1\n", "error: failed to load model\n"]
    with pytest.raises(EngineError, match="failed to load model"):
        wcpp.transcribe("tiny", samples(), [], StubJob(), lambda *a, **k: None)


def test_cancel_kills_the_process(wcpp, popen):
    put_model(wcpp)
    with pytest.raises(Cancelled):
        wcpp.transcribe("tiny", samples(), [], StubJob(cancel_after=1), lambda *a, **k: None)
    assert popen.last.killed


def test_requires_binary_and_downloaded_model(wcpp, popen):
    with pytest.raises(EngineError, match="não está baixado"):
        wcpp.transcribe("tiny", samples(), [], StubJob(), lambda *a, **k: None)


# ---------- download do modelo ----------

class FakeResponse:
    def __init__(self, data, status=200, etag=None):
        self.data, self.status_code = data, status
        self.headers = {"content-length": str(len(data)), **({"x-linked-etag": f'"{etag}"'} if etag else {})}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, n):
        for i in range(0, len(self.data), n):
            yield self.data[i : i + n]


def test_model_download_verifies_etag_and_reports_progress(wcpp, monkeypatch):
    import requests

    data = b"modelo-ggml" * 200_000
    urls = []
    monkeypatch.setattr(requests, "get", lambda url, **kw: urls.append(url) or FakeResponse(data, etag=hashlib.sha256(data).hexdigest()))
    seen = []
    assert not wcpp.is_installed("tiny-q5_1")
    wcpp.ensure_model("tiny-q5_1", lambda d, t: seen.append((d, t)))
    assert wcpp.is_installed("tiny-q5_1") and urls == ["https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny-q5_1.bin"]
    assert seen[-1][0] == seen[-1][1] == len(data)


def test_model_download_rejects_bad_checksum_and_404(wcpp, monkeypatch):
    import requests

    data = b"x" * 2_000_000
    monkeypatch.setattr(requests, "get", lambda url, **kw: FakeResponse(data, etag="0" * 64))
    with pytest.raises(EngineError, match="corrompido"):
        wcpp.ensure_model("tiny")
    assert not wcpp.is_installed("tiny") and not os.path.exists(wcpp._file("tiny") + ".part")
    monkeypatch.setattr(requests, "get", lambda url, **kw: FakeResponse(b"", status=404))
    with pytest.raises(EngineError, match="não existe"):
        wcpp.ensure_model("tiny")


def test_catalog_and_models_endpoint(client_factory, wcpp):
    put_model(wcpp, "tiny")
    path = wcpp.configured_path
    c, _ = client_factory()  # create_app aplica o caminho das Configurações (vazio): reaplica o do teste
    wcpp.configured_path = path
    models = c.get("/api/models?engine=whisper-cpp").get_json()["models"]
    assert models[0]["name"] == "tiny" and models[0]["installed"]  # baixado vem primeiro
    assert any(m["name"] == "large-v3-turbo-q5_0" and not m["installed"] for m in models)
    assert engines.get("whisper-cpp").label == "Whisper Nativo" and "temporário" not in engines.get("whisper-cpp").description
