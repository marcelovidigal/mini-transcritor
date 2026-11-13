"""Motores de transcrição alternativos ao Ollama: OpenAI Whisper, faster-whisper e WhisperX.

Os pacotes são OPCIONAIS e importados só quando usados (o app base e o executável continuam leves). Cada motor sabe:
dizer se está instalado (e como instalar), listar o catálogo (whisper_models.txt) marcando o que já foi baixado,
baixar um modelo sob demanda e transcrever os trechos de fala devolvendo segmentos com timestamps reais.
"""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .envutil import clean_env
from .ffmpeg_manager import app_dir

CATALOG_FILE = "whisper_models.txt"
OLLAMA = "ollama"


def _interruptible(job, fn, *args, **kw):
    """Roda uma chamada longa que NÃO dá para interromper por dentro (PyTorch/CTranslate2: carregar modelo, reconhecer o arquivo inteiro,
    alinhar, diarizar) numa thread à parte e confere o cancelamento a cada 0,3 s: cancelado, o worker sai na hora (`Cancelled`) e a
    tela/fila ficam livres; o cálculo em si não pára e termina sozinho em segundo plano (o resultado é descartado)."""
    box: dict = {}

    def work():
        try:
            box["v"] = fn(*args, **kw)
        except BaseException as e:  # noqa: BLE001 - repassado ao chamador
            box["e"] = e

    t = threading.Thread(target=work, daemon=True, name="interruptible-engine-call")
    t.start()
    while t.is_alive():
        t.join(0.3)
        job.check_cancel()
    if "e" in box:
        raise box["e"]
    return box.get("v")


class EngineError(Exception):
    """Falha de um motor alternativo (pacote ausente, modelo indisponível, erro de execução)."""


@dataclass
class Seg:
    start: float  # segundos, absolutos no arquivo
    end: float
    text: str
    speaker: str | None = None  # rótulo bruto da diarização (ex.: SPEAKER_00)


# ---------------------------------------------------------------- catálogo
def parse_catalog(text: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    seen = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2 or not parts[0] or not parts[1] or " " in parts[1] or (parts[0], parts[1]) in seen:
            continue
        seen.add((parts[0], parts[1]))
        out.setdefault(parts[0], []).append(
            {"name": parts[1], "approx_size": parts[2] if len(parts) > 2 else "", "note": parts[3] if len(parts) > 3 else ""})
    return out


def load_catalog(path: str | None = None) -> dict[str, list[dict]]:
    paths = [path] if path else [os.path.join(app_dir(), CATALOG_FILE)] + (
        [os.path.join(sys._MEIPASS, CATALOG_FILE)] if getattr(sys, "_MEIPASS", None) else [])
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                data = parse_catalog(f.read())
            if data:
                return data
        except OSError:
            continue
    return {}


# ---------------------------------------------------------------- utilidades
DEVICES = ("cpu", "cuda", "xpu", "mps")
DEVICE_LABELS = {"cpu": "CPU", "cuda": "GPU NVIDIA ou AMD (CUDA/ROCm)", "xpu": "GPU Intel (XPU)", "mps": "Apple Silicon (Metal)"}
_AUTO_ORDER = ("cuda", "xpu", "mps", "cpu")
_DEVICE_HINTS = {
    "cuda": "precisa de GPU NVIDIA (CUDA) ou AMD (ROCm) e do PyTorch correspondente (ou do CTranslate2 com CUDA)",
    "xpu": "instale o PyTorch com suporte a Intel XPU (build +xpu) e os drivers da GPU Intel",
    "mps": "só existe no macOS com Apple Silicon e PyTorch com suporte a MPS",
    "cpu": "",
}
_detected: dict | None = None


def detect_devices(refresh: bool = False) -> dict[str, bool]:
    """Dispositivos utilizáveis nesta máquina (cacheado; importar o PyTorch demora alguns segundos)."""
    global _detected
    if _detected is not None and not refresh:
        return dict(_detected)
    found = {"cpu": True, "cuda": False, "xpu": False, "mps": False}
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            found["cuda"] = bool(torch.cuda.is_available())
            xpu = getattr(torch, "xpu", None)
            found["xpu"] = bool(xpu is not None and xpu.is_available())
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            found["mps"] = bool(mps is not None and mps.is_available())
        except Exception:
            pass
    if not found["cuda"] and importlib.util.find_spec("ctranslate2") is not None:
        try:
            import ctranslate2

            found["cuda"] = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            pass
    _detected = found
    return dict(found)


def pick_device(requested: str = "auto", supported=DEVICES, label: str = "") -> tuple[str, str | None]:
    """Escolhe onde rodar. `requested`: auto|cpu|cuda|xpu|mps (vazio/auto = variável MT_DEVICE, senão automático).

    `supported` = o que o motor sabe usar. Devolve (dispositivo, aviso). Pedido explícito de um dispositivo que a
    máquina não tem é erro; que só o motor não suporta cai para a CPU com aviso. No automático, a ordem é
    CUDA > XPU > Metal > CPU, e avisa se havia GPU que o motor não aproveita."""
    req = (requested or "").strip().lower()
    if req in ("", "auto"):
        req = os.environ.get("MT_DEVICE", "auto").strip().lower() or "auto"
    if req != "auto" and req not in DEVICES:
        raise EngineError(f"Dispositivo inválido: '{req}'. Use auto, cpu, cuda, xpu ou mps.")
    have = detect_devices()
    if req != "auto":
        if not have[req]:
            hint = _DEVICE_HINTS[req]
            raise EngineError(f"O dispositivo '{req}' ({DEVICE_LABELS[req]}) não está disponível nesta máquina" + (f": {hint}." if hint else "."))
        if req in supported:
            return req, None
        return "cpu", f"{label or 'Este motor'} não usa {DEVICE_LABELS[req]}: rodando na CPU."
    best = next(d for d in _AUTO_ORDER if have[d])
    chosen = next(d for d in _AUTO_ORDER if have[d] and d in supported)
    note = None
    if chosen != best:
        note = f"{label or 'Este motor'} não usa {DEVICE_LABELS[best]}: rodando na CPU."
    return chosen, note


def devices_status() -> dict:
    have = detect_devices()
    best, _ = pick_device("auto")
    return {"devices": [{"id": d, "label": DEVICE_LABELS[d], "available": have[d]} for d in DEVICES], "active": best}


def engine_plan(engine_id: str, requested: str = "auto") -> list[dict]:
    """Onde cada etapa do motor vai rodar nesta máquina, separado do hardware detectado:
    [{"stage", "device" (id ou None), "label", "why" (por que não é a melhor GPU, ou o erro)}]."""
    eng = get(engine_id)
    if not eng:
        return []
    if hasattr(eng, "plan"):  # motores que escolhem o hardware por conta própria (runtime C++, servidor remoto)
        return eng.plan(requested)
    have = detect_devices()
    best, _ = pick_device("auto")
    plan = []
    for stage, supported in eng.stages:
        usable = [d for d in DEVICES if have[d] and d in supported]
        if not usable:
            plan.append({"stage": stage, "device": None, "label": "indisponível",
                         "why": "precisa de " + " ou ".join(DEVICE_LABELS[d] for d in supported)})
            continue
        try:
            dev, note = pick_device(requested, supported, stage)
        except EngineError as e:
            plan.append({"stage": stage, "device": None, "label": "indisponível", "why": str(e)})
            continue
        why = ""
        if dev == "cpu" and best != "cpu" and best not in supported:
            why = f"este motor não usa {DEVICE_LABELS[best]}; só roda em: " + ", ".join(DEVICE_LABELS[d] for d in supported)
        elif note:
            why = note
        plan.append({"stage": stage, "device": dev, "label": DEVICE_LABELS[dev], "why": why})
    return plan


def release_memory() -> None:
    """Devolve à GPU a memória em cache do PyTorch (CUDA/XPU/Metal) depois de usar um modelo Whisper, para outro programa
    (o Ollama) poder usá-la. Sem PyTorch carregado, não faz nada."""
    import gc

    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    for name in ("cuda", "xpu", "mps"):
        mod = getattr(torch, name, None)
        try:
            if mod is not None and hasattr(mod, "empty_cache"):
                mod.empty_cache()
        except Exception:
            pass


def to_float(samples: np.ndarray) -> np.ndarray:
    return samples.astype(np.float32) / 32768.0


def _tqdm_reporter(progress):
    """Subclasse de tqdm que repassa (baixado, total) em bytes ao `progress` (usada pelo huggingface_hub)."""
    from tqdm.auto import tqdm

    class Reporter(tqdm):
        def update(self, n=1):
            r = super().update(n)
            if self.unit == "B" and self.total and progress:
                progress(self.n, self.total)
            return r

    return Reporter


def main_language(langs: list[str]) -> str | None:
    langs = [l for l in langs if l]
    return Counter(langs).most_common(1)[0][0] if langs else None


# ---------------------------------------------------------------- base
class Engine:
    id = ""
    label = ""
    description = ""
    module = ""  # nome de importação do pacote
    pip = ""  # comando de instalação
    supports_diarization = False
    # etapas que usam hardware e os dispositivos que cada uma sabe usar: (nome da etapa, dispositivos suportados)
    stages: tuple = (("Reconhecimento (fala → texto)", DEVICES),)

    def available(self) -> bool:
        return importlib.util.find_spec(self.module) is not None

    def session(self, model, *, device="auto", context="", job=None):
        """Sessão com o modelo carregado para transcrever fragmento a fragmento com prompt próprio (só o WhisperX tem)."""
        return None

    def status(self) -> dict:
        ok = self.available()
        return {"id": self.id, "label": self.label, "description": self.description, "available": ok,
                "hint": "" if ok else self.not_installed_message(),
                "supports_diarization": self.supports_diarization}

    def catalog(self) -> list[dict]:
        return load_catalog().get(self.id, [])

    def models(self) -> list[dict]:
        """Catálogo no mesmo formato da lista do Ollama; `installed` só é verdadeiro para o que já está no disco."""
        ok = self.available()
        out = []
        for c in self.catalog():
            installed = ok and self.is_installed(c["name"])
            out.append({"name": c["name"], "installed": installed, "recommended": True, "size": 0,
                        "approx_size": c["approx_size"], "note": c["note"]})
        return sorted(out, key=lambda m: not m["installed"])  # baixados primeiro; mantém a ordem do catálogo

    def knows(self, model: str) -> bool:
        return any(c["name"] == model for c in self.catalog())

    def is_installed(self, model: str) -> bool:
        raise NotImplementedError

    def ensure_model(self, model: str, progress=None) -> None:
        """Baixa o modelo. `progress(baixado, total)` em bytes."""
        raise NotImplementedError

    def transcribe(self, model: str, samples: np.ndarray, chunks, job, upd, *, diarize=False, hf_token="",
                   parallel=1, context="", device="auto") -> tuple[list[Seg], str | None]:
        """Devolve (segmentos com timestamps absolutos, idioma detectado)."""
        raise NotImplementedError

    def not_installed_message(self) -> str:
        return f"O motor {self.label} não está instalado. Instale com: {self.pip}"

    def _require(self):
        if not self.available():
            raise EngineError(self.not_installed_message())


# ---------------------------------------------------------------- OpenAI Whisper
class WhisperEngine(Engine):
    id = "openai-whisper"
    label = "OpenAI Whisper"
    description = "Whisper original (PyTorch). Timestamps por trecho de fala."
    module = "whisper"
    pip = "pip install openai-whisper"

    @staticmethod
    def cache_dir() -> str:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        return os.path.join(base, "whisper")

    def _url(self, model: str) -> str:
        import whisper

        url = getattr(whisper, "_MODELS", {}).get(model)
        if not url:
            raise EngineError(f"A versão instalada do openai-whisper não conhece o modelo '{model}'. Atualize: pip install -U openai-whisper")
        return url

    def _file(self, model: str) -> str:
        return os.path.join(self.cache_dir(), os.path.basename(self._url(model)))

    def is_installed(self, model: str) -> bool:
        try:
            return os.path.isfile(self._file(model))
        except EngineError:
            return False

    def ensure_model(self, model, progress=None):
        import requests

        self._require()
        url, target = self._url(model), self._file(model)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        expected = url.split("/")[-2]  # o caminho do arquivo oficial traz o SHA-256
        part = target + ".part"
        sha = hashlib.sha256()
        try:
            with requests.get(url, stream=True, timeout=(15, 60)) as r:
                r.raise_for_status()
                total, done = int(r.headers.get("content-length", 0)), 0
                with open(part, "wb") as f:
                    for block in r.iter_content(1 << 20):
                        f.write(block)
                        sha.update(block)
                        done += len(block)
                        if progress:
                            progress(done, total)
        except requests.RequestException as e:
            _rm(part)
            raise EngineError(f"Falha ao baixar o modelo '{model}': {e}")
        if len(expected) == 64 and sha.hexdigest() != expected:
            _rm(part)
            raise EngineError(f"O download do modelo '{model}' está corrompido (SHA-256 não confere). Tente novamente.")
        os.replace(part, target)

    def transcribe(self, model, samples, chunks, job, upd, *, diarize=False, hf_token="", parallel=1, context="", device="auto"):
        self._require()
        import whisper

        dev, note = pick_device(device, DEVICES, self.label)
        if note:
            job.warnings.append(note)
        upd(0, f"Carregando o modelo Whisper ({DEVICE_LABELS[dev]})")
        m = whisper.load_model(model, device=dev, download_root=self.cache_dir())
        segs: list[Seg] = []
        langs: list[str] = []
        lang = None
        for i, ch in enumerate(chunks, 1):
            job.check_cancel()
            upd(100 * (i - 1) / len(chunks), f"Transcrevendo trecho {i} de {len(chunks)}")
            res = m.transcribe(to_float(ch.samples), language=lang, fp16=(dev == "cuda"), temperature=0.0,  # fp16 só no CUDA (Metal/XPU: float32)
                               
                               condition_on_previous_text=False, verbose=None, initial_prompt=context or None)
            if not lang and res.get("text", "").strip():
                lang = res.get("language")  # fixa o idioma detectado no 1º trecho com fala
            if res.get("language"):
                langs.append(res["language"])
            for s in res.get("segments", []):
                t = (s.get("text") or "").strip()
                if t:
                    segs.append(Seg(ch.start + float(s["start"]), min(ch.start + float(s["end"]), ch.end), t))
        del m
        release_memory()
        return segs, main_language(langs)


# ---------------------------------------------------------------- faster-whisper
_FW_PATTERNS = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"]


class FasterWhisperEngine(Engine):
    id = "faster-whisper"
    label = "Faster Whisper"
    description = "Whisper em CTranslate2: mais rápido e leve que o original."
    stages = (("Reconhecimento (fala → texto)", ("cpu", "cuda")),)  # CTranslate2 não tem Metal nem Intel XPU
    module = "faster_whisper"
    pip = "pip install faster-whisper"

    def _repo(self, model: str) -> str:
        from faster_whisper.utils import _MODELS

        repo = _MODELS.get(model)
        if not repo:
            raise EngineError(f"A versão instalada do faster-whisper não conhece o modelo '{model}'. Atualize: pip install -U faster-whisper")
        return repo

    def is_installed(self, model: str) -> bool:
        try:
            from faster_whisper.utils import download_model

            download_model(model, local_files_only=True)
            return True
        except Exception:
            return False

    def ensure_model(self, model, progress=None):
        self._require()
        from huggingface_hub import snapshot_download

        try:
            snapshot_download(self._repo(model), allow_patterns=_FW_PATTERNS, tqdm_class=_tqdm_reporter(progress))
        except EngineError:
            raise
        except Exception as e:
            raise EngineError(f"Falha ao baixar o modelo '{model}': {e}")

    def transcribe(self, model, samples, chunks, job, upd, *, diarize=False, hf_token="", parallel=1, context="", device="auto"):
        self._require()
        from faster_whisper import WhisperModel

        dev, note = pick_device(device, ("cpu", "cuda"), self.label)  # CTranslate2 não tem Metal nem Intel XPU
        if note:
            job.warnings.append(note)
        upd(0, f"Carregando o modelo faster-whisper ({DEVICE_LABELS[dev]})")
        m = _interruptible(job, WhisperModel, model, device=dev, compute_type="float16" if dev == "cuda" else "int8")
        segs: list[Seg] = []
        langs: list[str] = []
        lang = None
        for i, ch in enumerate(chunks, 1):
            job.check_cancel()
            upd(100 * (i - 1) / len(chunks), f"Transcrevendo trecho {i} de {len(chunks)}")
            it, info = m.transcribe(to_float(ch.samples), language=lang, beam_size=5, temperature=0.0,
                                    condition_on_previous_text=False, vad_filter=False,
                                    initial_prompt=context or None)
            got = False
            for s in it:
                t = (s.text or "").strip()
                if t:
                    got = True
                    segs.append(Seg(ch.start + float(s.start), min(ch.start + float(s.end), ch.end), t))
            if got and not lang:
                lang = info.language
            if info.language and got:
                langs.append(info.language)
        return segs, main_language(langs)


# ---------------------------------------------------------------- WhisperX
class WhisperXSession:
    """Modelo WhisperX (faster-whisper em lotes + VAD) mantido na memória; o `initial_prompt` muda a cada fragmento."""

    def __init__(self, model: str, device: str, context: str = ""):
        import whisperx

        self.context, self.lang, self.device = context or "", None, device
        self.model = whisperx.load_model(model, device, compute_type="float16" if device == "cuda" else "int8")
        self._base = self.model.options

    def transcribe(self, samples, prompt: str = "") -> str:
        """Texto de um fragmento. `prompt` (termos + versão anterior) entra como initial_prompt, junto do contexto do usuário."""
        import dataclasses

        full = " ".join(p for p in (self.context, prompt) if p).strip()
        self.model.options = dataclasses.replace(self._base, initial_prompt=full[:600] or None)
        res = self.model.transcribe(to_float(samples), batch_size=2, language=self.lang)
        text = " ".join((s.get("text") or "").strip() for s in res.get("segments", [])).strip()
        if text and not self.lang:
            self.lang = res.get("language")  # fixa o idioma do 1º fragmento com fala
        return text

    def close(self):
        self.model = None
        release_memory()


class WhisperXEngine(FasterWhisperEngine):
    id = "whisperx"
    label = "WhisperX"
    description = ("Reconhece a fala com o faster-whisper e detecta onde há voz (VAD); depois alinha cada palavra ao áudio e, se você pedir, "
                   "identifica quem fala (diarização).")
    stages = (("Reconhecimento (fala → texto)", ("cpu", "cuda")),
              ("Alinhamento por palavra e diarização", DEVICES))  # estas duas rodam no PyTorch
    module = "whisperx"
    pip = "pip install whisperx"
    supports_diarization = True

    def available(self) -> bool:
        # o WhisperX depende do faster-whisper (modelos e verificação de cache vêm dele)
        return importlib.util.find_spec("whisperx") is not None and importlib.util.find_spec("faster_whisper") is not None

    def session(self, model, *, device="auto", context="", job=None):
        """Modelo carregado UMA vez para transcrever fragmento a fragmento, com `initial_prompt` próprio em cada chamada
        (laço de refinamento da opção Qualidade). Sem alinhamento nem diarização: só o texto interessa."""
        self._require()
        asr_dev, note = pick_device(device, ("cpu", "cuda"), "O reconhecimento do WhisperX")
        if note and job is not None:
            job.warnings.append(note)
        return WhisperXSession(model, asr_dev, context)

    def transcribe(self, model, samples, chunks, job, upd, *, diarize=False, hf_token="", parallel=1, context="", device="auto"):
        self._require()
        import whisperx

        if diarize and not hf_token:
            raise EngineError("A diarização exige um token do Hugging Face (Configurações ou HF_TOKEN) e a aceitação da licença do modelo pyannote (speaker-diarization-community-1 ou -3.1).")
        asr_dev, note = pick_device(device, ("cpu", "cuda"), "O reconhecimento do WhisperX")
        torch_dev, _ = pick_device(device, DEVICES, self.label)  # alinhamento e diarização rodam no PyTorch
        if note:
            job.warnings.append(note + (f" Alinhamento e diarização usarão {DEVICE_LABELS[torch_dev]}." if torch_dev != "cpu" else ""))
        audio = to_float(samples)
        job.check_cancel()
        upd(0, f"Carregando o modelo WhisperX ({DEVICE_LABELS[asr_dev]})")
        opts = {"asr_options": {"initial_prompt": context}} if context else {}
        m = _interruptible(job, whisperx.load_model, model, asr_dev, compute_type="float16" if asr_dev == "cuda" else "int8", **opts)
        job.check_cancel()
        upd(15, "Transcrevendo (VAD + lotes)")
        res = _interruptible(job, m.transcribe, audio, batch_size=max(1, parallel) * 4 if asr_dev == "cuda" else 2)
        lang = res.get("language")
        job.check_cancel()
        upd(60, "Alinhando as palavras ao áudio")
        def align_on(dev):
            am, meta = _interruptible(job, whisperx.load_align_model, language_code=lang, device=dev)
            return _interruptible(job, whisperx.align, res["segments"], am, meta, audio, dev, return_char_alignments=False)

        try:
            try:
                res = align_on(torch_dev)
            except Exception:
                if torch_dev == "cpu":
                    raise
                job.warnings.append(f"Alinhamento em {DEVICE_LABELS[torch_dev]} falhou; refeito na CPU.")
                res = align_on("cpu")
        except Exception as e:  # idioma sem modelo de alinhamento: segue com os timestamps do Whisper
            job.warnings.append(f"Alinhamento por palavra indisponível para '{lang}': {e}")
        if diarize:
            job.check_cancel()
            upd(80, "Identificando quem fala (diarização)")
            try:
                try:
                    from whisperx.diarize import DiarizationPipeline
                except ImportError:
                    DiarizationPipeline = whisperx.DiarizationPipeline
                try:
                    dia, used = _interruptible(job, _diarize, DiarizationPipeline, hf_token, torch_dev, audio)
                except Exception:
                    if torch_dev == "cpu":
                        raise
                    job.warnings.append(f"Diarização em {DEVICE_LABELS[torch_dev]} falhou; refeita na CPU.")
                    dia, used = _interruptible(job, _diarize, DiarizationPipeline, hf_token, "cpu", audio)
                res = whisperx.assign_word_speakers(dia, res)
                job.warnings.append(f"Diarização com o modelo {used}.")
            except Exception as e:
                raise EngineError(f"Falha na diarização: {e}. Confira o token e se a licença do modelo pyannote foi aceita no Hugging Face.")
        segs = []
        for s in res.get("segments", []):
            t = (s.get("text") or "").strip()
            if t and "start" in s and "end" in s:
                segs.append(Seg(float(s["start"]), float(s["end"]), t, s.get("speaker")))
        return segs, lang


# ---------------------------------------------------------------- whisper.cpp
class WhisperCppEngine(Engine):
    """whisper.cpp via executável `whisper-cli` (C/C++, sem Python nem PyTorch); modelos GGML do Hugging Face (baixados quando
    selecionados).

    Um único motor com dois executáveis, escolhidos pela máquina (`choose_backend`): **CUDA** (GPU NVIDIA, Windows x64) quando
    há GPU NVIDIA, senão **CPU**. Cada um fica em `<app>/whispercpp/cpu/` e `.../cuda/` e é baixado do release oficial ao
    gerar a legenda (modo local). GPU Intel/AMD (Vulkan) não tem binário pronto: exigiria compilar."""

    stages = (("Reconhecimento (fala → texto)", ("cpu", "cuda")),)
    id = "whisper-cpp"
    label = "Whisper Nativo"
    description = ("whisper.cpp: Whisper em C/C++, sem Python/PyTorch. Usa a GPU NVIDIA (CUDA) quando houver e, senão, a CPU; processa o áudio "
                   "inteiro numa chamada, com timestamps do próprio modelo. O executável (CPU ~8 MB; CUDA centenas de MB) é "
                   "baixado ao gerar a legenda, se ainda não existir.")
    pip = ""
    env_var = "WHISPER_CPP_PATH"  # executável da CPU; o da CUDA usa env_var_cuda
    env_var_cuda = "WHISPER_CPP_CUDA_PATH"
    HF_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{name}.bin"
    configured_path = ""  # definidos pelas Configurações (pasta ou executável)
    configured_cuda_path = ""
    BACKEND_LABELS = {"cpu": "CPU", "cuda": "GPU NVIDIA (CUDA)"}

    @staticmethod
    def _names() -> tuple:
        return ("whisper-cli.exe", "whisper-cli") if sys.platform == "win32" else ("whisper-cli",)

    @classmethod
    def locate(cls, path: str) -> str | None:
        """Acha o whisper-cli em `path` (executável ou pasta, olhando também Release/ e bin/)."""
        if not path:
            return None
        if os.path.isfile(path):
            return path
        for sub_dir in ("", "Release", "bin", os.path.join("build", "bin")):
            for n in cls._names():
                cand = os.path.join(path, sub_dir, n)
                if os.path.isfile(cand):
                    return cand
        return None

    def _paths(self, variant: str) -> tuple:
        from . import whispercpp_manager as wm

        if variant == "cuda":
            return (self.configured_cuda_path, os.environ.get(self.env_var_cuda, ""), wm.variant_dir("cuda"))
        return (self.configured_path, os.environ.get(self.env_var, ""), wm.variant_dir("cpu"))

    def _find(self, variant: str) -> str | None:
        """Ordem: caminho das Configurações → variável de ambiente → pasta da variante em `whispercpp/` → (CPU) PATH."""
        for p in self._paths(variant):
            found = self.locate(p)
            if found:
                return found
        if variant == "cpu":
            for n in self._names():
                found = shutil.which(n)
                if found:
                    return found
        return None

    def choose_backend(self, device: str = "auto") -> tuple[str, str]:
        """(backend, motivo): "cuda" se houver GPU NVIDIA e um executável CUDA (pronto ou baixável), senão "cpu".
        O motivo é vazio quando a escolha é a esperada; explica quando o pedido não pôde ser atendido."""
        from . import whispercpp_manager as wm

        req = (device or "").strip().lower()
        if req in ("", "auto"):
            req = os.environ.get("MT_DEVICE", "auto").strip().lower() or "auto"
        if req == "cpu":
            return "cpu", "escolhido em Configurações → Dispositivo"
        cuda_ok = bool(self._find("cuda")) or wm.supported("cuda")
        if wm.has_nvidia() and cuda_ok:
            return "cuda", ""
        if req == "cuda":
            return "cpu", "sem GPU NVIDIA (ou sem binário CUDA para este sistema): usando a CPU"
        if req in ("xpu", "mps"):
            return "cpu", "o whisper.cpp só tem binário pronto para CPU e NVIDIA/CUDA: usando a CPU"
        return "cpu", ""

    def find_binary(self, variant: str | None = None, device: str = "auto") -> str | None:
        return self._find(variant or self.choose_backend(device)[0])

    def installed(self, device: str = "auto") -> bool:
        return self.find_binary(device=device) is not None

    def can_auto_install(self, device: str = "auto") -> bool:
        from . import whispercpp_manager as wm

        return wm.supported(self.choose_backend(device)[0])

    def available(self) -> bool:
        """Selecionável: algum executável já existe ou o da CPU pode ser baixado automaticamente (o app baixa ao gerar)."""
        from . import whispercpp_manager as wm

        return self.installed() or self.can_auto_install() or bool(self._find("cpu")) or wm.supported("cpu")

    def status(self) -> dict:
        st = super().status()
        st["installed"] = self.installed()
        st["needs_download"] = not self.installed() and self.can_auto_install()
        if not st["available"]:
            st["hint"] = self.not_installed_message()
        return st

    def plan(self, requested: str = "auto") -> list[dict]:
        """Plano de hardware próprio: CUDA se houver GPU NVIDIA, senão CPU (não depende do PyTorch)."""
        stage = self.stages[0][0]
        backend, why = self.choose_backend(requested)
        if backend == "cpu" and not why:
            from . import whispercpp_manager as wm

            if not wm.has_nvidia():
                why = "sem GPU NVIDIA; GPU Intel/AMD exigiria compilar o whisper.cpp com Vulkan"
        return [{"stage": stage, "device": backend, "label": self.BACKEND_LABELS[backend], "why": why}]

    def ensure_binary(self, progress=None, device: str = "auto") -> str:
        """Baixa o executável do backend escolhido, se faltar (modo local). `progress(baixado, total)` em bytes."""
        from . import whispercpp_manager as wm

        variant = self.choose_backend(device)[0]
        found = self._find(variant)
        if found:
            return found
        try:
            return wm.install(variant, progress)
        except wm.ManagerError as e:
            raise EngineError(str(e))

    def not_installed_message(self) -> str:
        return ("Não há binário pronto do whisper.cpp para este sistema. Compile-o (github.com/ggml-org/whisper.cpp) e informe a "
                f"pasta em Configurações ou em {self.env_var}.")

    @staticmethod
    def models_dir() -> str:
        from .settings import data_dir

        return os.path.join(data_dir(), "whispercpp-models")

    def _file(self, model: str) -> str:
        return os.path.join(self.models_dir(), f"ggml-{model}.bin")

    def is_installed(self, model: str) -> bool:
        f = self._file(model)
        return os.path.isfile(f) and os.path.getsize(f) > 1_000_000

    def ensure_model(self, model, progress=None):
        import requests

        target = self._file(model)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        part = target + ".part"
        sha = hashlib.sha256()
        try:
            with requests.get(self.HF_URL.format(name=model), stream=True, timeout=(15, 60), allow_redirects=True) as r:
                if r.status_code == 404:
                    raise EngineError(f"O modelo 'ggml-{model}.bin' não existe no repositório do whisper.cpp (nome do catálogo desatualizado?).")
                r.raise_for_status()
                expected = (r.headers.get("x-linked-etag") or "").strip('"')  # o Hugging Face publica o SHA-256 dos arquivos LFS
                total, done = int(r.headers.get("content-length", 0)), 0
                with open(part, "wb") as f:
                    for block in r.iter_content(1 << 20):
                        f.write(block)
                        sha.update(block)
                        done += len(block)
                        if progress:
                            progress(done, total)
        except EngineError:
            _rm(part)
            raise
        except requests.RequestException as e:
            _rm(part)
            raise EngineError(f"Falha ao baixar o modelo '{model}': {e}")
        if re.fullmatch(r"[0-9a-f]{64}", expected) and sha.hexdigest() != expected:
            _rm(part)
            raise EngineError(f"O download do modelo '{model}' está corrompido (SHA-256 não confere). Tente novamente.")
        os.replace(part, target)

    def transcribe(self, model, samples, chunks, job, upd, *, diarize=False, hf_token="", parallel=1, context="", device="auto"):
        backend, why = self.choose_backend(device)
        exe = self._find(backend)
        if not exe:
            raise EngineError("O executável do whisper.cpp ainda não foi baixado: " + self.not_installed_message())
        job.warnings.append(f"whisper.cpp executado em {self.BACKEND_LABELS[backend]}" + (f" ({why})." if why else "."))
        if not self.is_installed(model):
            raise EngineError(f"O modelo ggml-{model}.bin não está baixado.")
        from . import audio

        with tempfile.TemporaryDirectory(prefix="mt_wcpp_") as tmp:
            wav, prefix = os.path.join(tmp, "audio.wav"), os.path.join(tmp, "saida")
            with open(wav, "wb") as f:
                f.write(audio.to_wav_bytes(samples))
            cmd = [exe, "-m", self._file(model), "-f", wav, "-oj", "-of", prefix, "-l", "auto", "-pp",
                   "-t", str(max(1, (os.cpu_count() or 4) // 2))]
            if backend == "cpu":
                cmd.append("-ng")  # backend CPU (por escolha ou por falta de GPU NVIDIA): sem GPU
            if context:
                cmd += ["--prompt", " ".join(context.split())[:500]]
            upd(0, f"Executando o whisper.cpp ({self.BACKEND_LABELS[backend]})")
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                    errors="replace", env=clean_env())
            tail: list[str] = []
            try:
                for line in proc.stderr:
                    job.check_cancel()
                    m = re.search(r"progress\s*=\s*(\d+)\s*%", line)
                    if m:
                        upd(min(int(m.group(1)), 99), "Transcrevendo (whisper.cpp)")
                    elif line.strip():
                        tail = (tail + [line.strip()])[-6:]
                proc.wait()
            except BaseException:
                proc.kill()
                proc.wait()
                raise
            if proc.returncode != 0:
                raise EngineError("O whisper.cpp falhou (código %s): %s" % (proc.returncode, " | ".join(tail[-3:]) or "sem detalhes"))
            try:
                with open(prefix + ".json", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError) as e:
                raise EngineError(f"O whisper.cpp terminou, mas a saída JSON não pôde ser lida: {e}")
        segs = []
        for item in data.get("transcription", []):
            text = (item.get("text") or "").strip()
            off = item.get("offsets") or {}
            if text and "from" in off and "to" in off:
                segs.append(Seg(off["from"] / 1000.0, off["to"] / 1000.0, text))
        return segs, (data.get("result") or {}).get("language")


# ---------------------------------------------------------------- NeMo-Speech.cpp (NVIDIA)
_HF = "https://huggingface.co"
_ATOMIC = f"{_HF}/AtomicChat/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF/resolve/dbd06c5cad7a2f33c8e06be2fb3ef814dbd9aa59"
_ENCODER = {"name": "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-Encoder-F16.gguf", "size": 1670410080,
            "sha256": "4221029c8a565c4a80c507ffa760c716e595acf663f9f4273ce69910050e7ba2"}
# modelos do motor: arquivo(s) GGUF com tamanho e SHA-256 exatos (conferidos no download); o 1º arquivo é o que se passa ao --model.
# O Transcribe-30B é um par: decodificador + codificador (o decodificador procura o codificador pelo nome, na mesma pasta).
NEMO_MODELS: dict[str, dict] = {
    "nemotron-3.5-asr-streaming-0.6b": {
        "kind": "asr", "window": 3600.0,
        "files": [{"name": "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf", "size": 741548352,
                   "sha256": "a5c435f294eea8f88ce68dd27b8c3bfea7f777cb2fbba04fcd30eaa555f429ae",
                   "url": f"{_HF}/nvidia/nemotron-3.5-asr-streaming-0.6b/resolve/1c8deaecc64b91f034d73e08dd8b64625eb3395d/"
                          "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"}]},
    "nemotron-3.5-transcribe-30b-a4b-q2_0_m": {
        "kind": "salm", "window": 3000.0,
        "files": [{"name": "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-AD-Q2_0_M.gguf", "size": 13572208800,
                   "sha256": "764dc3aedbc800a96346e188d74f56fc4d0932299e4771df1bc2e1c952ceed81",
                   "url": f"{_ATOMIC}/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-AD-Q2_0_M.gguf"},
                  {**_ENCODER, "url": f"{_ATOMIC}/{_ENCODER['name']}"}]},
    "nemotron-3.5-transcribe-30b-a4b-q2_0_l": {
        "kind": "salm", "window": 3000.0,
        "files": [{"name": "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-AD-Q2_0_L.gguf", "size": 15547824288,
                   "sha256": "67eee66a7bb205867e3f9b7c0944e287fc4d2661db2280030e1c26e6ebdea7b4",
                   "url": f"{_ATOMIC}/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-AD-Q2_0_L.gguf"},
                  {**_ENCODER, "url": f"{_ATOMIC}/{_ENCODER['name']}"}]},
}
MAX_CPS_WORD_TIMES = 20.0  # caracteres por segundo aceitos num trecho montado com tempos por palavra
_SENTENCE_END = (".", "?", "!", "…", "。")


def words_to_segments(words: list[dict], offset: float = 0.0, max_gap: float = 0.8, max_dur: float = 12.0, max_chars: int = 200) -> list[Seg]:
    """Agrupa palavras com tempo (`word`, `start`, `end`) em trechos de legenda: quebra em pausa longa, fim de frase depois de
    ~2,5 s, duração máxima ou tamanho máximo. `offset` = início da janela no arquivo (segundos)."""
    segs: list[Seg] = []
    cur: list[dict] = []

    def flush():
        if cur:
            text = " ".join(str(w.get("word", "")).strip() for w in cur).strip()
            if text:
                segs.append(Seg(offset + float(cur[0]["start"]), offset + max(float(cur[-1]["end"]), float(cur[0]["start"]) + 0.2), text))
            cur.clear()

    for w in words:
        if "start" not in w or "end" not in w or not str(w.get("word", "")).strip():
            continue
        if cur:
            gap = float(w["start"]) - float(cur[-1]["end"])
            dur = float(cur[-1]["end"]) - float(cur[0]["start"])
            chars = sum(len(str(x.get("word", ""))) + 1 for x in cur)
            ended = str(cur[-1].get("word", "")).rstrip().endswith(_SENTENCE_END)
            if gap > max_gap or (ended and dur >= 2.5) or dur >= max_dur or chars >= max_chars:
                flush()
        cur.append(w)
    flush()
    return segs


def parse_nemo_json(data: dict, offset: float = 0.0) -> tuple[list[Seg], str | None]:
    """Resultado `--json` do nemo-speech → (trechos com tempo absoluto, idioma). Aceita palavras com tempo ou trechos prontos."""
    segs: list[Seg] = []
    if data.get("words"):
        segs = words_to_segments(data["words"], offset)
    elif data.get("segments"):
        for s in data["segments"]:
            t = str(s.get("text", "")).strip()
            if t and "start" in s and "end" in s:
                segs.append(Seg(offset + float(s["start"]), offset + float(s["end"]), t, None))
    elif str(data.get("text", "")).strip():
        segs = [Seg(offset, offset + max(float(data.get("duration") or 0.0), 0.2), str(data["text"]).strip())]
    langs = [l for l in (data.get("languages") or []) if l and l != "und"]
    return segs, (langs[0].split("-")[0].lower() if langs else None)


def window_chunks(chunks, max_seconds: float) -> list[tuple[float, float]]:
    """Agrupa os trechos de fala em janelas contíguas de até `max_seconds` (limite do modelo: 1 h), cortando entre trechos."""
    wins: list[tuple[float, float]] = []
    start = end = None
    for ch in chunks:
        if start is None:
            start, end = ch.start, ch.end
        elif ch.end - start > max_seconds:
            wins.append((start, end))
            start, end = ch.start, ch.end
        else:
            end = ch.end
    if start is not None:
        wins.append((start, end))
    return wins


def chunk_segments(data: dict, chunk) -> list[Seg]:
    """Trechos de legenda de UM trecho de fala (resultado `--json` do nemo-speech para o arquivo desse trecho). O tempo vem do VAD do app
    (`chunk.start`..`chunk.end`); os tempos por palavra do modelo só são usados se passarem numa checagem de sanidade (em ordem, dentro do
    trecho e sem saltos grandes: o cabeçote CTC do Transcribe-30B às vezes empacota palavras num instante e joga outras vários segundos
    adiante). Reprovado, o texto inteiro fica com o intervalo do trecho, como nos demais motores."""
    text = str(data.get("text", "")).strip()
    if not text:
        return []
    words = [w for w in (data.get("words") or []) if "start" in w and "end" in w and str(w.get("word", "")).strip()]
    dur = chunk.end - chunk.start
    ok = bool(words) and words[0]["start"] >= -0.05 and words[-1]["end"] <= dur + 0.6
    if ok:
        prev_end = -1.0
        for w in words:
            if w["start"] < prev_end - 0.05 or w["end"] < w["start"] or w["start"] - max(prev_end, 0.0) > 6.0:
                ok = False
                break
            prev_end = w["end"]
    if ok and (words[-1]["end"] - words[0]["start"]) < 0.25 * dur and len(words) > 8:  # tudo espremido num canto do trecho
        ok = False
    if ok:
        segs = words_to_segments(words, chunk.start)
        # densidade: fala natural fica em ~12–18 caracteres/s; o CTC às vezes amontoa a última frase no fim do trecho (ex.: 97 caracteres
        # em 1,9 s) e o filtro de alucinação a descartaria. Acima do limite, vale o intervalo inteiro do trecho.
        if segs and all(len(sg.text) / max(sg.end - sg.start, 0.1) <= MAX_CPS_WORD_TIMES for sg in segs):
            return [Seg(max(sg.start, chunk.start), min(sg.end, chunk.end), sg.text) for sg in segs]
    return [Seg(chunk.start, chunk.end, text)]


def format_elapsed(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:d}:{s % 60:02d}"


def _download(url: str, target: str, size: int, sha256: str, progress=None, base: int = 0, total: int = 0, what: str = "") -> None:
    """Baixa `url` para `target` (retomável, via arquivo .part) e confere tamanho e SHA-256 antes de aceitar."""
    import requests

    os.makedirs(os.path.dirname(target), exist_ok=True)
    part = target + ".part"
    have = os.path.getsize(part) if os.path.isfile(part) else 0
    if have > size:
        _rm(part)
        have = 0
    try:
        if have < size:
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, stream=True, timeout=(15, 60), allow_redirects=True, headers=headers) as r:
                if r.status_code == 404:
                    raise EngineError(f"O arquivo {what or os.path.basename(target)} não existe mais no Hugging Face.")
                if have and r.status_code != 206:  # servidor ignorou o Range: recomeça
                    have = 0
                r.raise_for_status()
                with open(part, "ab" if have else "wb") as f:
                    done = have
                    for block in r.iter_content(1 << 20):
                        f.write(block)
                        done += len(block)
                        if progress:
                            progress(base + done, total or size)
    except EngineError:
        raise
    except requests.RequestException as e:
        raise EngineError(f"Falha ao baixar {what or os.path.basename(target)}: {e} (o download continua de onde parou na próxima tentativa).")
    if os.path.getsize(part) != size:
        raise EngineError(f"O download de {what or os.path.basename(target)} terminou incompleto ({os.path.getsize(part)} de {size} bytes). Tente novamente.")
    sha = hashlib.sha256()
    with open(part, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            sha.update(block)
    if sha.hexdigest() != sha256:
        _rm(part)
        raise EngineError(f"O download de {what or os.path.basename(target)} está corrompido (SHA-256 não confere). Tente novamente.")
    os.replace(part, target)


class NemoSpeechCppEngine(Engine):
    """NeMo-Speech.cpp (`nemo-speech`): runtime C++ oficial da NVIDIA (ggml) para os modelos Nemotron, sem Python nem PyTorch.
    Modelos GGUF: Nemotron 3.5 Transcribe 30B-A4B (2 quantizações) e Nemotron 3.5 ASR Streaming 0.6B. Roda em CPU, GPU NVIDIA
    (CUDA) e qualquer GPU com Vulkan (Intel Arc, AMD). O executável é baixado do release oficial ao gerar a legenda."""

    id = "nemo-speech-cpp"
    label = "NemoSpeech"
    description = ("Modelos de fala da NVIDIA em C++ (sem Python/PyTorch), com tempo por palavra: Transcribe 30B-A4B (duas quantizações "
                   "Q2, para GPU de 16 GB ou mais) e ASR Streaming 0,6 B (leve). O executável (~10–500 MB, conforme a GPU) e o modelo "
                   "são baixados ao gerar a legenda. Qualquer arquivo de até 1 h por janela; mais longos são divididos.")
    stages = (("Reconhecimento (fala → texto)", ("cpu", "cuda", "xpu")),)  # Vulkan atende a GPU Intel (xpu) e AMD; CUDA, a NVIDIA
    env_var = "NEMO_SPEECH_PATH"
    configured_path = ""
    pip = ""

    @staticmethod
    def models_dir() -> str:
        from .settings import data_dir

        return os.path.join(data_dir(), "nemocpp-models")

    def spec(self, model: str) -> dict:
        s = NEMO_MODELS.get(model)
        if not s:
            raise EngineError(f"Modelo desconhecido para o NeMo-Speech.cpp: {model}")
        return s

    def model_dir(self, model: str) -> str:
        return os.path.join(self.models_dir(), model)

    def model_file(self, model: str) -> str:
        return os.path.join(self.model_dir(model), self.spec(model)["files"][0]["name"])

    def find_binary(self) -> str | None:
        from . import nemocpp_manager as nm

        for p in (self.configured_path, os.environ.get(self.env_var, ""), nm.install_dir()):
            found = nm.locate(p)
            if found:
                return found
        return shutil.which("nemo-speech")

    def installed(self, device: str = "auto") -> bool:
        return self.find_binary() is not None

    def can_auto_install(self) -> bool:
        from . import nemocpp_manager as nm

        os_name, arch = nm.machine()
        return (os_name, arch) in {("windows", "x86_64"), ("linux", "x86_64"), ("linux", "aarch64"), ("macos", "aarch64"), ("macos", "x86_64")}

    def available(self) -> bool:
        return self.installed() or self.can_auto_install()

    def status(self) -> dict:
        st = super().status()
        st["installed"] = self.installed()
        st["needs_download"] = not self.installed() and self.can_auto_install()
        return st

    def ensure_binary(self, progress=None, device: str = "auto") -> str:
        from . import nemocpp_manager as nm

        found = self.find_binary()
        if found:
            return found
        try:
            return nm.install(progress)
        except nm.ManagerError as e:
            raise EngineError(str(e))

    def not_installed_message(self) -> str:
        return ("Não há binário pronto do NeMo-Speech.cpp para este sistema. Compile-o (github.com/NVIDIA/NeMo-Speech.cpp) e informe a "
                f"pasta em Configurações ou em {self.env_var}.")

    def is_installed(self, model: str) -> bool:
        s = NEMO_MODELS.get(model)
        if not s:
            return False
        d = self.model_dir(model)
        return all(os.path.isfile(os.path.join(d, f["name"])) and os.path.getsize(os.path.join(d, f["name"])) == f["size"] for f in s["files"])

    def ensure_model(self, model, progress=None):
        s = self.spec(model)
        d = self.model_dir(model)
        total = sum(f["size"] for f in s["files"])
        base = 0
        for f in s["files"]:
            target = os.path.join(d, f["name"])
            if not (os.path.isfile(target) and os.path.getsize(target) == f["size"]):
                _download(f["url"], target, f["size"], f["sha256"], progress, base, total, f["name"])
            base += f["size"]

    @staticmethod
    def _device_flag(device: str) -> list[str]:
        """Pedido da tela → `--device` do nemo-speech. Automático deixa o próprio runtime escolher (GPU se houver)."""
        req = (device or "").strip().lower()
        if req in ("", "auto"):
            req = os.environ.get("MT_DEVICE", "auto").strip().lower() or "auto"
        return {"cpu": ["--device", "cpu"], "cuda": ["--device", "cuda:0"], "xpu": ["--device", "vulkan:0"]}.get(req, [])

    def plan(self, requested: str = "auto") -> list[dict]:
        """Plano de hardware próprio: quem escolhe a GPU é o runtime (CUDA > Vulkan > CPU), não o PyTorch."""
        stage = self.stages[0][0]
        flag = self._device_flag(requested)
        if flag[-1:] == ["cpu"]:
            return [{"stage": stage, "device": "cpu", "label": "CPU", "why": "escolhido em Configurações → Dispositivo"}]
        from . import nemocpp_manager as nm

        backend = nm.installed_backend() or nm.preferred_backends()[0]
        label = {"cuda": "GPU NVIDIA (CUDA)", "vulkan": "GPU via Vulkan (Intel Arc, AMD ou NVIDIA)", "metal": "Apple Silicon (Metal)", "cpu": "CPU"}[backend]
        return [{"stage": stage, "device": "cpu" if backend == "cpu" else "gpu", "label": label,
                 "why": "o próprio NeMo-Speech.cpp escolhe a GPU; sem GPU compatível cai para a CPU"}]

    def transcribe(self, model, samples, chunks, job, upd, *, diarize=False, hf_token="", parallel=1, context="", device="auto"):
        from . import audio, langid

        exe = self.find_binary()
        if not exe:
            raise EngineError("O executável do NeMo-Speech.cpp ainda não foi baixado: " + self.not_installed_message())
        s = self.spec(model)
        if not self.is_installed(model):
            raise EngineError(f"O modelo {model} não está baixado.")
        if s["kind"] == "salm":
            return self._transcribe_salm(exe, model, chunks, job, upd, context, device)
        wins = window_chunks(chunks, s["window"])
        if context and s["kind"] == "asr":
            job.warnings.append("O modelo ASR Streaming 0,6 B não usa o contexto/glossário (só o Transcribe 30B aceita instruções).")
        segs: list[Seg] = []
        langs: list[str] = []
        with tempfile.TemporaryDirectory(prefix="mt_nemocpp_") as tmp:
            for i, (w0, w1) in enumerate(wins, 1):
                job.check_cancel()
                part = samples[int(w0 * audio.SAMPLE_RATE): int(w1 * audio.SAMPLE_RATE)]
                wav = os.path.join(tmp, f"janela{i}.wav")
                with open(wav, "wb") as f:
                    f.write(audio.to_wav_bytes(part))
                cmd = [exe, "transcribe", wav, "--model", self.model_file(model), "--json"] + self._device_flag(device)
                if s["kind"] == "asr":
                    cmd += ["--language", "auto"]
                elif context:
                    cmd += ["--instruction", "Transcribe with punctuation and capitalization. Context terms: "
                            + " ".join(context.split())[:500] + "."]
                label = "Transcrevendo (NeMo-Speech.cpp)" + (f", janela {i} de {len(wins)}" if len(wins) > 1 else "")
                data = self._run(cmd, job, lambda el, lbl=label, k=i: upd(100 * (k - 1) / len(wins), f"{lbl} · {format_elapsed(el)}"))
                got, lang = parse_nemo_json(data, w0)
                segs += got
                if lang:
                    langs.append(lang)
        lang = main_language(langs) or langid.detect(" ".join(sg.text for sg in segs)[:4000])
        return segs, lang

    CONCURRENCY = 4  # trechos decodificados juntos (o modelo é carregado uma vez; em lote fica ~3x mais rápido que uma janela longa)

    def _transcribe_salm(self, exe, model, chunks, job, upd, context, device):
        """Transcribe-30B: cada trecho de fala do VAD vira um arquivo; UMA execução do CLI (modelo carregado uma vez) processa a pasta com
        `--concurrency`. Vantagens sobre uma janela longa: bem mais rápido, sem texto inventado em música/ruído (só vai fala) e tempo
        confiável (vem do VAD)."""
        from . import audio, langid

        with tempfile.TemporaryDirectory(prefix="mt_nemocpp_") as tmp:
            indir, outdir = os.path.join(tmp, "in"), os.path.join(tmp, "out")
            os.makedirs(indir)
            for i, ch in enumerate(chunks):
                with open(os.path.join(indir, f"c{i:06d}.wav"), "wb") as f:
                    f.write(audio.to_wav_bytes(ch.samples))
            cmd = [exe, "transcribe", indir, "--model", self.model_file(model), "--format", "json", "--output-dir", outdir,
                   "--concurrency", str(self.CONCURRENCY)] + self._device_flag(device)
            if context:
                cmd += ["--instruction", "Transcribe with punctuation and capitalization. Context terms: " + " ".join(context.split())[:500] + "."]
            prog = {"k": 0, "n": len(chunks)}

            def tick(el):
                upd(100 * prog["k"] / max(prog["n"], 1), f"Transcrevendo (NeMo-Speech.cpp) · trecho {prog['k']} de {prog['n']} · {format_elapsed(el)}")

            self._run(cmd, job, tick, prog, want_json=False)
            segs: list[Seg] = []
            for i, ch in enumerate(chunks):
                try:
                    with open(os.path.join(outdir, f"c{i:06d}.json"), encoding="utf-8-sig") as f:
                        data = json.load(f)
                except (OSError, ValueError):
                    job.warnings.append(f"Trecho {i + 1}: o NeMo-Speech.cpp não devolveu resultado; ficou sem texto.")
                    continue
                segs += chunk_segments(data, ch)
        return segs, langid.detect(" ".join(sg.text for sg in segs)[:4000])  # o modelo não informa o idioma: detectado pelo texto

    def _run(self, cmd: list[str], job, tick, prog: dict | None = None, want_json: bool = True) -> dict:
        """Executa o nemo-speech numa subprocesso que PODE ser morto no cancelamento (ao contrário das chamadas nativas dentro do
        Python). Devolve o JSON de stdout."""
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=clean_env(),
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        out: dict = {}

        def drain(name, stream):
            if name == "err" and prog is not None:  # linhas "[k/n] arquivo -> saída": progresso real do modo pasta
                buf = []
                for raw in stream:
                    buf.append(raw)
                    m = re.match(rb"\[(\d+)/(\d+)\]", raw)
                    if m:
                        prog["k"], prog["n"] = int(m.group(1)), int(m.group(2))
                out[name] = b"".join(buf)
            else:
                out[name] = stream.read()

        readers = [threading.Thread(target=drain, args=(n, st), daemon=True) for n, st in (("out", proc.stdout), ("err", proc.stderr))]
        for r in readers:
            r.start()
        t0 = time.monotonic()
        try:
            while proc.poll() is None:
                job.check_cancel()
                tick(time.monotonic() - t0)
                time.sleep(0.4)
            for r in readers:
                r.join(5)
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        stdout = (out.get("out") or b"").decode("utf-8", errors="replace")
        stderr = (out.get("err") or b"").decode("utf-8", errors="replace")
        if proc.returncode != 0:
            msg = stdout.strip() or stderr.strip()
            try:
                msg = json.loads(stdout)["error"]["message"]
            except Exception:
                pass
            raise EngineError(f"O NeMo-Speech.cpp falhou (código {proc.returncode}): {' | '.join(msg.splitlines()[-3:])[:400] or 'sem detalhes'}")
        if not want_json:
            return {}
        try:
            return json.loads(stdout)
        except ValueError as e:
            raise EngineError(f"O NeMo-Speech.cpp terminou, mas a saída JSON não pôde ser lida: {e}")


# ---------------------------------------------------------------- Nemotron Transcribe NVFP4 (servidor vLLM)
class NemotronVllmEngine(Engine):
    """Nemotron 3.5 Transcribe 30B-A4B **NVFP4** por um servidor vLLM (API compatível com a da OpenAI). Segundo o cartão do modelo,
    o NVFP4 só roda em GPU NVIDIA **Blackwell** e é servido pelo vLLM com o plugin do NeMo Speech (não por `nemo_asr.from_pretrained`).
    O app não instala nada disso: ele é cliente de um servidor que você já tenha de pé (Configurações → endereço do servidor)."""

    id = "nemotron-vllm"
    hidden_unless_available = True  # fora da lista da tela enquanto nenhum servidor vLLM responder
    label = "NemotronNvfp4"
    description = ("O modelo oficial NVFP4 da NVIDIA servido por vLLM (exige GPU NVIDIA Blackwell, Linux/WSL2). O app envia cada trecho "
                   "ao servidor que você informou em Configurações e monta a legenda com o texto devolvido.")
    stages = (("Reconhecimento (no servidor vLLM)", ("cuda",)),)
    MODEL = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"
    DEFAULT_URL = "http://127.0.0.1:8000/v1"
    configured_url = ""
    pip = ""
    PROMPT = "Transcribe the following audio."

    def url(self) -> str:
        return (self.configured_url or os.environ.get("NEMOTRON_VLLM_URL", "") or self.DEFAULT_URL).rstrip("/")

    def served(self) -> list[str]:
        """Modelos que o servidor serve agora (vazio se não responde)."""
        import requests

        try:
            r = requests.get(self.url() + "/models", timeout=2)
            r.raise_for_status()
            return [m.get("id", "") for m in r.json().get("data", [])]
        except Exception:
            return []

    def available(self) -> bool:
        return bool(self.served())

    def not_installed_message(self) -> str:
        return (f"Nenhum servidor vLLM respondeu em {self.url()}. Este modelo (NVFP4) só roda em GPU NVIDIA Blackwell, servido pelo vLLM "
                "com o plugin do NeMo Speech (instruções no cartão do modelo, no Hugging Face). Informe o endereço em Configurações.")

    def status(self) -> dict:
        st = super().status()
        st["needs_download"] = False
        return st

    def is_installed(self, model: str) -> bool:
        return model in self.served()

    def ensure_model(self, model, progress=None):
        if model not in self.served():
            raise EngineError(f"O servidor em {self.url()} não está servindo o modelo {model}.")

    def plan(self, requested: str = "auto") -> list[dict]:
        return [{"stage": self.stages[0][0], "device": "gpu", "label": f"GPU Blackwell no servidor ({self.url()})",
                 "why": "o app não usa a GPU desta máquina: só envia o áudio ao servidor"}]

    def transcribe(self, model, samples, chunks, job, upd, *, diarize=False, hf_token="", parallel=1, context="", device="auto"):
        import base64

        import requests

        from . import audio, langid

        prompt = self.PROMPT if not context else "Transcribe with punctuation and capitalization. Context terms: " + " ".join(context.split())[:500] + "."
        segs: list[Seg] = []
        for i, ch in enumerate(chunks, 1):
            job.check_cancel()
            upd(100 * (i - 1) / len(chunks), f"Transcrevendo trecho {i} de {len(chunks)} (vLLM)")
            b64 = base64.b64encode(audio.to_wav_bytes(ch.samples)).decode()
            body = {"model": model, "temperature": 0,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b64}"}}]}],
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}, "skip_special_tokens": False}}
            body.update(body.pop("extra_body"))  # o vLLM aceita esses campos no nível raiz
            try:
                r = requests.post(self.url() + "/chat/completions", json=body, timeout=(5, 300))
                r.raise_for_status()
                text = (r.json()["choices"][0]["message"].get("content") or "").strip()
            except (requests.RequestException, KeyError, IndexError, ValueError) as e:
                raise EngineError(f"O servidor vLLM falhou no trecho {i}: {e}")
            if text:
                segs.append(Seg(ch.start, ch.end, text))
        return segs, langid.detect(" ".join(sg.text for sg in segs)[:4000])


DIARIZATION_MODELS = ("pyannote/speaker-diarization-3.1", None)  # None = o padrão do WhisperX (community-1, mais novo e também restrito)


def _diarizer(cls, token: str, device: str, model_name: str | None = None):
    """Cria o DiarizationPipeline do WhisperX: a 3.8 usa `token=`; versões anteriores usavam `use_auth_token=`."""
    last = None
    extra = {"model_name": model_name} if model_name else {}
    for kw in ("token", "use_auth_token"):
        try:
            return cls(**{kw: token}, device=device, **extra)
        except TypeError as e:  # parâmetro desconhecido nesta versão
            last = e
    raise last


def _diarize(cls, token: str, device: str, audio):
    """Roda a diarização tentando os modelos em ordem (env WHISPERX_DIARIZATION_MODEL, 3.1, padrão do WhisperX): cada um
    exige aceitar a licença no Hugging Face, então o que o usuário já aceitou é usado. Devolve (resultado, modelo)."""
    candidates = [m for m in (os.environ.get("WHISPERX_DIARIZATION_MODEL") or None,) if m] + list(DIARIZATION_MODELS)
    errors = []
    for name in candidates:
        try:
            return _diarizer(cls, token, device, name)(audio), name or "padrão do WhisperX"
        except Exception as e:
            errors.append(f"{name or 'padrão'}: {str(e).splitlines()[0][:160]}")
    raise RuntimeError("nenhum modelo de diarização acessível (" + " | ".join(errors) + ")")


def _rm(path: str):
    try:
        os.remove(path)
    except OSError:
        pass


ENGINES: dict[str, Engine] = {e.id: e for e in (WhisperEngine(), FasterWhisperEngine(), WhisperXEngine(), WhisperCppEngine(),
                                                 NemoSpeechCppEngine(), NemotronVllmEngine())}


def get(engine_id: str) -> Engine | None:
    return ENGINES.get(engine_id)


def all_status() -> list[dict]:
    """Todos os motores para a tela; o Ollama (padrão) vem primeiro e é sempre 'disponível' (o preparo cuida dele)."""
    first = {"id": OLLAMA, "label": "Ollama (multimodal)", "available": True, "hint": "", "supports_diarization": False,
             "description": "Modelos multimodais com áudio rodando no Ollama (padrão). Timestamps aproximados."}
    from .quality_engine import MODES

    modes = [{"id": mid, "label": m["label"], "available": True, "hint": "", "supports_diarization": False, "description": m["description"],
              "fixed_models": {"generator": m["generator"], "reviewer": m["reviewer"]}} for mid, m in MODES.items()]
    # motores `hidden_unless_available` (cliente vLLM: exige um servidor Blackwell) só aparecem quando há servidor respondendo
    return [first] + modes + [e.status() for e in ENGINES.values() if not (getattr(e, "hidden_unless_available", False) and not e.available())]



def check_request(engine_id: str, model: str, diarize: bool, hf_token: str, installed_only: bool = False) -> str | None:
    """Valida motor/modelo/diarização de um pedido de legenda. Devolve a mensagem de erro (ou None se estiver tudo certo)."""
    eng = ENGINES.get(engine_id)
    if not eng:
        return "Motor de transcrição inválido."
    if not eng.available():
        return eng.not_installed_message()
    if not eng.knows(model):
        return "O modelo selecionado não consta no catálogo deste motor."
    if installed_only and not eng.is_installed(model):
        return "O modelo selecionado não está disponível neste servidor."
    if diarize:
        if not eng.supports_diarization:
            return f"O motor {eng.label} não faz diarização. Use o WhisperX."
        if not hf_token:
            return "A diarização exige um token do Hugging Face (Configurações) e a aceitação da licença do modelo pyannote."
    return None
