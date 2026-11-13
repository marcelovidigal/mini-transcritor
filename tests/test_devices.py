import importlib.machinery
import sys
import types

import numpy as np
import pytest

from transcritor import engines
from transcritor.audio import Chunk
from transcritor.engines import EngineError

ALL = {"cpu": True, "cuda": True, "xpu": True, "mps": True}
ONLY_CPU = {"cpu": True, "cuda": False, "xpu": False, "mps": False}
XPU = {"cpu": True, "cuda": False, "xpu": True, "mps": False}
MPS = {"cpu": True, "cuda": False, "xpu": False, "mps": True}


@pytest.fixture
def have(monkeypatch):
    monkeypatch.delenv("MT_DEVICE", raising=False)

    def set_(d):
        monkeypatch.setattr(engines, "detect_devices", lambda refresh=False: dict(d))

    return set_


class StubJob:
    def __init__(self):
        self.warnings = []

    def check_cancel(self):
        pass


def pkg(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def chunks():
    return [Chunk(0.0, 5.0, np.zeros(16000, dtype=np.int16))]


# ---------- escolha do dispositivo ----------

def test_auto_prefers_cuda_then_xpu_then_mps_then_cpu(have):
    for avail, expected in ((ALL, "cuda"), (XPU, "xpu"), (MPS, "mps"), (ONLY_CPU, "cpu")):
        have(avail)
        assert engines.pick_device("auto") == (expected, None)


def test_engine_without_support_falls_back_to_cpu_with_note(have):
    have(XPU)
    dev, note = engines.pick_device("auto", ("cpu", "cuda"), "faster-whisper")
    assert dev == "cpu" and "Intel (XPU)" in note and "faster-whisper" in note
    have(MPS)
    assert engines.pick_device("auto", ("cpu", "cuda"))[0] == "cpu"
    have(ONLY_CPU)
    assert engines.pick_device("auto", ("cpu", "cuda")) == ("cpu", None)  # nada a avisar: não há GPU


def test_explicit_device_unavailable_is_an_error_with_a_hint(have):
    have(ONLY_CPU)
    with pytest.raises(EngineError, match="xpu"):
        engines.pick_device("xpu")
    with pytest.raises(EngineError, match="build \\+xpu"):
        engines.pick_device("xpu")
    with pytest.raises(EngineError, match="Apple Silicon"):
        engines.pick_device("mps")
    with pytest.raises(EngineError, match="inválido"):
        engines.pick_device("tpu")
    have(XPU)
    assert engines.pick_device("xpu") == ("xpu", None)
    assert engines.pick_device("cpu") == ("cpu", None)
    assert engines.pick_device("xpu", ("cpu", "cuda"))[0] == "cpu"  # existe, mas o motor não usa


def test_env_var_is_used_when_setting_is_auto(have, monkeypatch):
    have(ALL)
    monkeypatch.setenv("MT_DEVICE", "cpu")
    assert engines.pick_device("auto") == ("cpu", None) and engines.pick_device("") == ("cpu", None)
    assert engines.pick_device("xpu") == ("xpu", None)  # a configuração explícita vence a variável


def test_detect_devices_without_torch_is_cpu_only(monkeypatch):
    monkeypatch.setattr(engines.importlib.util, "find_spec", lambda name: None)
    assert engines.detect_devices(refresh=True) == ONLY_CPU
    monkeypatch.undo()
    engines.detect_devices(refresh=True)  # recoloca o cache real


def test_detect_devices_reads_torch(monkeypatch):
    torch = pkg(monkeypatch, "torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.xpu = types.SimpleNamespace(is_available=lambda: True)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    assert engines.detect_devices(refresh=True) == {"cpu": True, "cuda": False, "xpu": True, "mps": False}
    monkeypatch.undo()
    engines.detect_devices(refresh=True)


# ---------- motores ----------

def test_openai_whisper_uses_requested_device_and_fp16_only_on_cuda(monkeypatch, have):
    seen = {}

    class Model:
        def transcribe(self, audio, **kw):
            seen["kw"] = kw
            return {"text": "hello", "language": "en", "segments": [{"start": 0, "end": 1, "text": "hello"}]}

    pkg(monkeypatch, "whisper", load_model=lambda name, device, download_root: seen.update(device=device) or Model())
    eng = engines.get("openai-whisper")
    for avail, dev, fp16 in ((ALL, "cuda", True), (XPU, "xpu", False), (MPS, "mps", False), (ONLY_CPU, "cpu", False)):
        have(avail)
        job = StubJob()
        segs, lang = eng.transcribe("tiny", None, chunks(), job, lambda *a, **k: None, device="auto")
        assert seen["device"] == dev and seen["kw"]["fp16"] is fp16 and lang == "en" and segs[0].text == "hello"
        assert not job.warnings


def test_faster_whisper_never_receives_xpu_or_mps(monkeypatch, have):
    seen = {}

    class WM:
        def __init__(self, model, device, compute_type):
            seen.update(device=device, compute_type=compute_type)

        def transcribe(self, audio, **kw):
            seg = types.SimpleNamespace(text="hi", start=0.0, end=1.0)
            return iter([seg]), types.SimpleNamespace(language="en")

    pkg(monkeypatch, "faster_whisper", WhisperModel=WM)
    eng = engines.get("faster-whisper")
    have(XPU)
    job = StubJob()
    eng.transcribe("tiny", None, chunks(), job, lambda *a, **k: None)
    assert seen == {"device": "cpu", "compute_type": "int8"} and any("Intel (XPU)" in w for w in job.warnings)
    have(ALL)
    eng.transcribe("tiny", None, chunks(), StubJob(), lambda *a, **k: None)
    assert seen == {"device": "cuda", "compute_type": "float16"}


def test_whisperx_splits_asr_on_cpu_and_torch_parts_on_xpu_with_cpu_fallback(monkeypatch, have):
    calls = {"asr": None, "align": []}

    class Model:
        def transcribe(self, audio, batch_size):
            return {"language": "en", "segments": [{"start": 0.0, "end": 2.0, "text": "hello"}]}

    def load_model(name, device, compute_type, **kw):
        calls["asr"] = (device, compute_type)
        return Model()

    def load_align_model(language_code, device):
        calls["align"].append(device)
        if device == "xpu":
            raise RuntimeError("operador não suportado em XPU")
        return object(), {}

    pkg(monkeypatch, "faster_whisper")
    pkg(monkeypatch, "whisperx", load_model=load_model, load_align_model=load_align_model,
        align=lambda segs, am, meta, audio, dev, return_char_alignments: {"segments": segs})
    have(XPU)
    job = StubJob()
    segs, lang = engines.get("whisperx").transcribe("tiny", np.zeros(16000, dtype=np.int16), chunks(), job, lambda *a, **k: None)
    assert calls["asr"] == ("cpu", "int8") and calls["align"] == ["xpu", "cpu"]
    assert segs[0].text == "hello" and lang == "en"
    assert any("Alinhamento e diarização usarão" in w for w in job.warnings) and any("refeito na CPU" in w for w in job.warnings)


def test_forced_unavailable_device_fails_the_job_clearly(monkeypatch, have):
    pkg(monkeypatch, "whisper", load_model=lambda *a, **k: None)
    have(ONLY_CPU)
    with pytest.raises(EngineError, match="xpu"):
        engines.get("openai-whisper").transcribe("tiny", None, chunks(), StubJob(), lambda *a, **k: None, device="xpu")


# ---------- API e configurações ----------

def test_devices_endpoint_and_setting(client_factory, have):
    have(XPU)
    c, _ = client_factory()
    body = c.get("/api/devices").get_json()
    assert body["active"] == "xpu" and {d["id"]: d["available"] for d in body["devices"]} == XPU
    assert c.post("/api/settings", json={"device": "xpu"}).get_json()["device"] == "xpu"
    assert c.post("/api/settings", json={"device": "tpu"}).status_code == 400
    assert c.get("/api/devices").get_json()["selected"] == "xpu"
    c.post("/api/settings", json={"device": "auto"})


def test_diarizer_factory_supports_both_whisperx_apis():
    class New:
        def __init__(self, model_name=None, token=None, device="cpu"):
            self.args = ("token", token, device)

    class Old:
        def __init__(self, model_name=None, use_auth_token=None, device="cpu"):
            self.args = ("use_auth_token", use_auth_token, device)

    assert engines._diarizer(New, "hf_x", "xpu").args == ("token", "hf_x", "xpu")
    assert engines._diarizer(Old, "hf_x", "cpu").args == ("use_auth_token", "hf_x", "cpu")
