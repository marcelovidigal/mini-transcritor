import hashlib
import io
import os
import zipfile

import pytest

from transcritor import engines
from transcritor import whispercpp_manager as wm
from transcritor.engines import EngineError, WhisperCppEngine
from transcritor.jobs import Job
from transcritor.pipeline import _ensure_engine_model

from .test_whispercpp import FakePopen, StubJob, popen, put_model, samples  # noqa: F401  (fixture popen)

# nomes reais de assets observados nos releases do whisper.cpp (2026)
V192 = ["whisper-bin-ubuntu-arm64.tar.gz", "whisper-bin-ubuntu-x64.tar.gz", "whisper-bin-Win32.zip", "whisper-bin-x64.zip",
        "whisper-blas-bin-Win32.zip", "whisper-blas-bin-x64.zip", "whisper-cublas-11.8.0-bin-x64.zip", "whisper-cublas-12.4.0-bin-x64.zip"]
B5454 = ["whisper-b5454-xcframework.zip", "whisper-bin-ubuntu-arm64.tar.gz", "whisper-bin-ubuntu-x64.tar.gz", "whisper-bin-win-cpu-arm64.zip",
         "whisper-bin-win-cuda-11.8.0-x64.zip", "whisper-bin-win-cuda-12.4.0-x64.zip", "whisper-bin-win-cuda-13.4-arm64.zip",
         "whisper-bin-win-opencl-adreno-arm64.zip"]


def assets(names):
    return [{"name": n, "browser_download_url": "https://x.test/" + n, "size": 10} for n in names]


# ---------- escolha do asset ----------

def test_pick_asset_cpu_and_cuda_for_windows_x64():
    assert wm.pick_asset(assets(V192), "cpu", "windows", "x64")["name"] == "whisper-bin-x64.zip"
    assert wm.pick_asset(assets(V192), "cuda", "windows", "x64")["name"] == "whisper-cublas-12.4.0-bin-x64.zip"  # 12.x preferido ao 11.8
    assert wm.pick_asset(assets(B5454), "cuda", "windows", "x64")["name"] == "whisper-bin-win-cuda-12.4.0-x64.zip"
    assert wm.pick_asset(assets(B5454), "cpu", "windows", "x64") is None  # esse release não traz o CPU x64: o gerenciador tenta o anterior


def test_pick_asset_other_platforms():
    assert wm.pick_asset(assets(V192), "cpu", "linux", "x64")["name"] == "whisper-bin-ubuntu-x64.tar.gz"
    assert wm.pick_asset(assets(V192), "cpu", "linux", "arm64")["name"] == "whisper-bin-ubuntu-arm64.tar.gz"
    assert wm.pick_asset(assets(B5454), "cpu", "windows", "arm64")["name"] == "whisper-bin-win-cpu-arm64.zip"
    assert wm.pick_asset(assets(V192), "cuda", "linux", "x64") is None  # sem CUDA pronto fora do Windows x64
    assert wm.pick_asset(assets(V192), "cpu", "macos", "arm64") is None
    assert wm.pick_asset(assets(V192), "cuda", "windows", "arm64") is None


def test_cuda_prefers_12_then_13_then_11():
    names = ["whisper-cublas-11.8.0-bin-x64.zip", "whisper-cublas-13.1.0-bin-x64.zip", "whisper-cublas-12.2.0-bin-x64.zip", "whisper-cublas-12.4.0-bin-x64.zip"]
    assert wm.pick_asset(assets(names), "cuda", "windows", "x64")["name"] == "whisper-cublas-12.4.0-bin-x64.zip"
    assert wm.pick_asset(assets(names[:2]), "cuda", "windows", "x64")["name"] == "whisper-cublas-13.1.0-bin-x64.zip"


# ---------- instalação (GitHub simulado) ----------

def make_zip(inner="Release/whisper-cli.exe"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(inner, "binario")
        z.writestr(os.path.dirname(inner) + "/ggml.dll", "dll")
    return buf.getvalue()


class FakeHttp:
    def __init__(self, releases, payload):
        self.releases, self.payload, self.urls = releases, payload, []

    def get(self, url, **kw):
        self.urls.append(url)
        if url == wm.API:
            return FakeJson(self.releases)
        return FakeDownload(self.payload)


class FakeJson:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


class FakeDownload:
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
        yield self.data


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(wm, "machine", lambda: ("windows", "x64"))


def test_install_uses_the_newest_release_that_has_the_asset_and_checks_digest(windows, tmp_path):
    payload = make_zip()
    releases = [{"tag_name": "b5454", "assets": [dict(a, digest="") for a in assets(B5454)]},  # sem CPU x64: pula
                {"tag_name": "v1.9.2", "assets": [dict(a, digest="sha256:" + hashlib.sha256(payload).hexdigest()) for a in assets(V192)]}]
    http = FakeHttp(releases, payload)
    seen = []
    exe = wm.install("cpu", lambda d, t: seen.append((d, t)), http=http, dest=str(tmp_path / "cpu"))
    assert os.path.basename(exe) == "whisper-cli.exe" and os.path.isfile(exe) and os.path.isfile(os.path.join(tmp_path, "cpu", "ggml.dll"))
    assert http.urls[-1].endswith("whisper-bin-x64.zip") and seen[-1][0] == seen[-1][1] == len(payload)
    # CUDA vem do release mais novo que o tem
    exe = wm.install("cuda", http=http, dest=str(tmp_path / "cuda"))
    assert http.urls[-1].endswith("whisper-bin-win-cuda-12.4.0-x64.zip") and os.path.isfile(exe)
    assert os.path.isdir(tmp_path / "cpu") and os.path.isdir(tmp_path / "cuda")  # duas versões lado a lado


def test_install_rejects_corrupted_download_and_missing_cli(windows, tmp_path):
    payload = make_zip()
    rel = [{"tag_name": "v", "assets": [dict(a, digest="sha256:" + "0" * 64) for a in assets(V192)]}]
    with pytest.raises(wm.ManagerError, match="corrompido"):
        wm.install("cpu", http=FakeHttp(rel, payload), dest=str(tmp_path / "x"))
    assert not os.path.exists(tmp_path / "x")
    empty = make_zip("Release/outra-coisa.exe")
    rel = [{"tag_name": "v", "assets": assets(V192)}]
    with pytest.raises(wm.ManagerError, match="não contém o whisper-cli"):
        wm.install("cpu", http=FakeHttp(rel, empty), dest=str(tmp_path / "y"))


def test_find_release_asset_errors_are_explained(windows, monkeypatch):
    with pytest.raises(wm.ManagerError, match="Nenhum release"):
        wm.find_release_asset("cpu", FakeHttp([{"tag_name": "v", "assets": assets(["outro.zip"])}], b""))
    monkeypatch.setattr(wm, "machine", lambda: ("macos", "arm64"))
    with pytest.raises(wm.ManagerError, match="compile"):
        wm.find_release_asset("cpu", FakeHttp([], b""))


# ---------- um motor, dois executáveis ----------

def _fake_exe(folder):
    folder.mkdir(parents=True, exist_ok=True)
    exe = folder / WhisperCppEngine._names()[0]
    exe.write_text("")
    return str(exe)


def test_one_engine_finds_each_backend_in_its_own_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(wm, "app_dir", lambda: str(tmp_path))
    monkeypatch.delenv("WHISPER_CPP_PATH", raising=False)
    monkeypatch.delenv("WHISPER_CPP_CUDA_PATH", raising=False)
    monkeypatch.setattr(engines.shutil, "which", lambda n: None)
    eng = WhisperCppEngine()
    assert eng.id == "whisper-cpp" and "whisper-cpp-cuda" not in engines.ENGINES and "temporário" not in eng.label
    cpu, cuda = _fake_exe(tmp_path / "whispercpp" / "cpu"), _fake_exe(tmp_path / "whispercpp" / "cuda")
    assert eng.find_binary("cpu") == cpu and eng.find_binary("cuda") == cuda
    monkeypatch.setenv("WHISPER_CPP_CUDA_PATH", str(tmp_path / "outro"))
    other = _fake_exe(tmp_path / "outro")
    assert eng.find_binary("cuda") == other and eng.find_binary("cpu") == cpu


def test_backend_choice_follows_the_hardware_and_the_device_setting(monkeypatch, tmp_path):
    monkeypatch.setattr(wm, "app_dir", lambda: str(tmp_path))  # sem executáveis instalados
    monkeypatch.delenv("WHISPER_CPP_CUDA_PATH", raising=False)
    monkeypatch.delenv("MT_DEVICE", raising=False)
    monkeypatch.setattr(wm, "machine", lambda: ("windows", "x64"))
    eng = WhisperCppEngine()
    monkeypatch.setattr(wm, "has_nvidia", lambda: True)
    assert eng.choose_backend("auto") == ("cuda", "") and eng.choose_backend("cuda") == ("cuda", "")
    assert eng.choose_backend("cpu")[0] == "cpu"
    assert eng.choose_backend("xpu")[0] == "cuda"  # GPU NVIDIA presente: o pedido de outra GPU não impede
    monkeypatch.setenv("MT_DEVICE", "cpu")
    assert eng.choose_backend("auto")[0] == "cpu"
    monkeypatch.delenv("MT_DEVICE")
    monkeypatch.setattr(wm, "has_nvidia", lambda: False)
    assert eng.choose_backend("auto") == ("cpu", "")
    backend, why = eng.choose_backend("cuda")
    assert backend == "cpu" and "NVIDIA" in why
    assert eng.choose_backend("xpu")[0] == "cpu" and "CPU e NVIDIA" in eng.choose_backend("xpu")[1]
    monkeypatch.setattr(wm, "has_nvidia", lambda: True)
    monkeypatch.setattr(wm, "machine", lambda: ("linux", "x64"))  # sem binário CUDA pronto fora do Windows
    assert eng.choose_backend("auto")[0] == "cpu"


def test_plan_names_the_backend(monkeypatch):
    monkeypatch.delenv("MT_DEVICE", raising=False)
    monkeypatch.setattr(wm, "machine", lambda: ("windows", "x64"))
    monkeypatch.setattr(wm, "has_nvidia", lambda: True)
    p = engines.engine_plan("whisper-cpp")[0]
    assert p["device"] == "cuda" and p["label"] == "GPU NVIDIA (CUDA)" and p["why"] == ""
    monkeypatch.setattr(wm, "has_nvidia", lambda: False)
    p = engines.engine_plan("whisper-cpp")[0]
    assert p["device"] == "cpu" and p["label"] == "CPU" and "Vulkan" in p["why"]
    p = engines.engine_plan("whisper-cpp", "cpu")[0]
    assert p["device"] == "cpu" and "Configurações" in p["why"]


def test_status_and_catalog(monkeypatch, tmp_path):
    monkeypatch.setattr(wm, "app_dir", lambda: str(tmp_path))
    monkeypatch.setattr(engines.shutil, "which", lambda n: None)
    monkeypatch.setattr(wm, "machine", lambda: ("windows", "x64"))
    monkeypatch.setattr(wm, "has_nvidia", lambda: False)
    st = WhisperCppEngine().status()
    assert st["available"] and st["needs_download"] and not st["installed"]
    monkeypatch.setattr(wm, "machine", lambda: ("macos", "arm64"))  # nenhum binário pronto
    st = WhisperCppEngine().status()
    assert not st["available"] and "whisper.cpp" in st["hint"]
    assert "whisper-cpp-cuda" not in engines.load_catalog() and engines.load_catalog()["whisper-cpp"]


def test_transcribe_uses_gpu_unless_cpu_is_requested(monkeypatch, tmp_path, popen):
    monkeypatch.setenv("MINITRANSCRITOR_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("MT_DEVICE", raising=False)
    monkeypatch.setattr(wm, "machine", lambda: ("windows", "x64"))
    eng = engines.get("whisper-cpp")
    monkeypatch.setattr(eng, "configured_path", str(tmp_path / "cpu"))
    monkeypatch.setattr(eng, "configured_cuda_path", str(tmp_path / "cuda"))
    _fake_exe(tmp_path / "cpu")
    cuda_exe = _fake_exe(tmp_path / "cuda")
    monkeypatch.setattr(wm, "has_nvidia", lambda: True)
    put_model(eng)
    job = StubJob()
    eng.transcribe("tiny", samples(), [], job, lambda *a, **k: None)
    assert "-ng" not in popen.last.cmd and popen.last.cmd[0] == cuda_exe
    assert any("GPU NVIDIA (CUDA)" in w for w in job.warnings)  # o log diz qual backend foi usado
    job = StubJob()
    eng.transcribe("tiny", samples(), [], job, lambda *a, **k: None, device="cpu")
    assert "-ng" in popen.last.cmd and popen.last.cmd[0] != cuda_exe and any("CPU" in w for w in job.warnings)
    monkeypatch.setattr(wm, "has_nvidia", lambda: False)  # sem GPU NVIDIA: CPU sem erro
    eng.transcribe("tiny", samples(), [], StubJob(), lambda *a, **k: None)
    assert "-ng" in popen.last.cmd


# ---------- pipeline: baixa o executável antes do modelo ----------

def test_pipeline_downloads_the_binary_then_the_model(monkeypatch, tmp_path):
    class Dual(WhisperCppEngine):
        id = "dual-test"
        calls = []

        def __init__(self):
            self.have_bin = False

        def installed(self, device="auto"):
            return self.have_bin

        def ensure_binary(self, progress=None, device="auto"):
            progress(5, 10)
            progress(10, 10)
            self.have_bin = True
            self.calls.append("bin")
            return "x"

        def _require(self):
            assert self.have_bin  # só depois de baixar o executável

        def knows(self, model):
            return True

        def is_installed(self, model):
            return "model" in self.calls

        def ensure_model(self, model, progress=None):
            progress(1, 1)
            self.calls.append("model")

    eng = Dual()
    job = Job(id="j", path="p", language="pt-BR", model="tiny", engine="dual-test")
    assert _ensure_engine_model(job, eng) > 0 and eng.calls == ["bin", "model"]
    # modo servidor: nunca baixa
    srv = Job(id="s", path="p", language="pt-BR", model="tiny", engine="dual-test", owner="a@b.c")
    with pytest.raises(EngineError, match="não está instalado neste servidor"):
        _ensure_engine_model(srv, Dual())


def test_settings_cuda_path_reaches_the_single_engine(client_factory, tmp_path):
    c, _ = client_factory()
    _fake_exe(tmp_path / "cu")
    r = c.post("/api/settings", json={"whispercpp_cuda_path": str(tmp_path / "cu")})
    assert r.get_json()["ok"] and engines.get("whisper-cpp").configured_cuda_path == str(tmp_path / "cu")
    c.post("/api/settings", json={"whispercpp_cuda_path": ""})
    assert engines.get("whisper-cpp").configured_cuda_path == ""
