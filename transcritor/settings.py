"""Configurações persistidas em JSON no perfil do usuário."""
import json
import os
import sys
import threading

from . import ffmpeg_manager
from .catalog import has_explicit_tag
from .config import normalize_host

DEFAULT_BASE_MODEL = "gemma4:e4b"


def data_dir() -> str:
    if os.environ.get("MINITRANSCRITOR_HOME"):
        return os.environ["MINITRANSCRITOR_HOME"]
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "MiniTranscritor")
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Application Support/MiniTranscritor")
    return os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"), "mini-transcritor")


def _defaults() -> dict:
    try:
        chunk = float(os.environ.get("CHUNK_SECONDS", 12))
    except ValueError:
        chunk = 12.0
    return {
        "ollama_host": normalize_host(os.environ.get("OLLAMA_HOST", "")),
        "base_model": DEFAULT_BASE_MODEL,
        "auto_pull": True,
        "ffmpeg_path": "",
        "chunk_seconds": chunk,
        "hf_token": "",
        "whispercpp_path": "",
        "whispercpp_cuda_path": "",
        "nemocpp_path": "",
        "nemotron_vllm_url": "",
        "last_dir": "",  # última pasta aberta no navegador de arquivos (ponto de partida da próxima vez)
        "device": "auto",
        "parallel": 2,
        "translate_batch": 8,
    }


class SettingsError(ValueError):
    pass


class Settings:
    def __init__(self, path: str | None = None):
        self.path = path or os.path.join(data_dir(), "settings.json")
        self._lock = threading.Lock()
        self._data = _defaults()
        try:
            with open(self.path, encoding="utf-8") as f:
                self.update(json.load(f), save=False)
        except (OSError, ValueError):
            pass  # arquivo ausente/corrompido: usa os padrões

    def as_dict(self) -> dict:
        with self._lock:
            return dict(self._data)

    def __getitem__(self, key):
        with self._lock:
            return self._data[key]

    def update(self, new: dict, save: bool = True) -> dict:
        """Valida e aplica as chaves conhecidas; levanta SettingsError se algum valor for inválido."""
        clean = {}
        if "ollama_host" in new:
            host = normalize_host(str(new["ollama_host"]))
            if len(host.split("//", 1)[1].strip("/")) == 0:
                raise SettingsError("Endereço do Ollama inválido.")
            clean["ollama_host"] = host
        if "base_model" in new:
            model = str(new["base_model"]).strip()
            if not model or " " in model:
                raise SettingsError("Informe um nome de modelo válido (ex.: gemma4:e4b).")
            if not has_explicit_tag(model):
                raise SettingsError("Use a tag completa do modelo, sem 'latest' (ex.: gemma4:e4b, nemotron3:33b-q4_K_M).")
            clean["base_model"] = model
        if "auto_pull" in new:
            clean["auto_pull"] = bool(new["auto_pull"])
        if "ffmpeg_path" in new:
            path = str(new["ffmpeg_path"] or "").strip()
            if path and not ffmpeg_manager.find_in_path_or_dir(path):
                raise SettingsError("Nesse caminho não foram encontrados ffmpeg e ffprobe. Informe a pasta que contém "
                                    "os dois (ou o próprio ffmpeg).")
            clean["ffmpeg_path"] = path
        if "chunk_seconds" in new:
            try:
                sec = float(new["chunk_seconds"])
            except (TypeError, ValueError):
                raise SettingsError("Duração dos trechos inválida.")
            if not 5 <= sec <= 28:
                raise SettingsError("A duração dos trechos deve estar entre 5 e 28 segundos.")
            clean["chunk_seconds"] = sec
        for key in ("whispercpp_path", "whispercpp_cuda_path"):
            if key in new:
                from .engines import WhisperCppEngine

                wp = str(new[key] or "").strip().strip('"')
                if wp and not WhisperCppEngine.locate(wp):
                    raise SettingsError("Nesse caminho não foi encontrado o whisper-cli (whisper.cpp). Informe a pasta que o contém (ou o próprio executável).")
                clean[key] = wp
        if "nemocpp_path" in new:
            from . import nemocpp_manager

            np_ = str(new["nemocpp_path"] or "").strip().strip('"')
            if np_ and not nemocpp_manager.locate(np_):
                raise SettingsError("Nesse caminho não foi encontrado o nemo-speech (NeMo-Speech.cpp). Informe a pasta que o contém (ou o próprio executável).")
            clean["nemocpp_path"] = np_
        if "nemotron_vllm_url" in new:
            url = str(new["nemotron_vllm_url"] or "").strip().rstrip("/")
            if url and not url.lower().startswith(("http://", "https://")):
                raise SettingsError("O endereço do servidor vLLM deve começar com http:// ou https:// (ex.: http://127.0.0.1:8000/v1).")
            clean["nemotron_vllm_url"] = url
        if "last_dir" in new:
            clean["last_dir"] = str(new["last_dir"] or "").strip()[:1024]
        if "device" in new:
            dev = str(new["device"] or "auto").strip().lower()
            if dev not in ("auto", "cpu", "cuda", "xpu", "mps"):
                raise SettingsError("Dispositivo inválido. Use auto, cpu, cuda, xpu ou mps.")
            clean["device"] = dev
        if "hf_token" in new:
            tok = str(new["hf_token"] or "").strip()
            if " " in tok:
                raise SettingsError("Token do Hugging Face inválido.")
            clean["hf_token"] = tok
        for key, lo, hi, label in (("parallel", 1, 4, "Chamadas simultâneas"), ("translate_batch", 1, 16, "Falas por lote de tradução")):
            if key in new:
                try:
                    val = int(new[key])
                except (TypeError, ValueError):
                    raise SettingsError(f"{label}: valor inválido.")
                if not lo <= val <= hi:
                    raise SettingsError(f"{label} deve estar entre {lo} e {hi}.")
                clean[key] = val
        with self._lock:
            self._data.update(clean)
            data = dict(self._data)
        if save:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        return data
