"""Preparo na inicialização: FFmpeg, Ollama (iniciar se preciso) e modelo base."""
import shutil
import sys
import threading
import time

from . import ffmpeg_manager as fm
from . import ollama_manager as om
from .ollama_client import OllamaClient, OllamaError

FFMPEG_HELP = {
    "darwin": "Instale com:  brew install ffmpeg",
    "win32": "Baixe em https://ffmpeg.org/download.html ou instale com:  winget install Gyan.FFmpeg",
}.get(sys.platform, "Instale com o gerenciador de pacotes (ex.: sudo apt install ffmpeg)")


class SetupManager:
    def __init__(self, settings, client: OllamaClient, process: om.OllamaProcess | None = None):
        self.settings = settings
        self.client = client
        self.process = process or om.OllamaProcess()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.state = {"status": "idle", "message": "", "progress": 0.0, "detail": ""}
        self.remote_wait_attempts = 30
        self.can_auto_install = sys.platform == "win32"  # instalador do Ollama

    # --- estado -------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return {**self.state, "base_model": self.settings["base_model"],
                    "can_auto_install": self.can_auto_install, "download_page": om.DOWNLOAD_PAGE,
                    "ffmpeg": fm.active_description()}

    def _set(self, status=None, message=None, progress=None, detail=None):
        with self._lock:
            for k, v in (("status", status), ("message", message), ("progress", progress), ("detail", detail)):
                if v is not None:
                    self.state[k] = v

    def mark_ready(self):
        self._set("ready", "Pronto", 100, "")

    # --- execução -----------------------------------------------------
    def run(self) -> bool:
        """Inicia o preparo em segundo plano. Devolve False se já houver um em andamento."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
            self.state.update(status="checking", message="Verificando o ambiente…", progress=0.0, detail="")
            self._thread = threading.Thread(target=self._safe_run, daemon=True)
            self._thread.start()
            return True

    def _safe_run(self):
        try:
            self._run()
        except OllamaError as e:
            self._set("error", str(e))
        except Exception as e:  # nunca deixar o preparo preso em "checking"
            self._set("error", f"Erro inesperado no preparo: {e}")

    def _run(self):
        if not self._ensure_ffmpeg():
            return

        host = self.settings["ollama_host"]
        self.client.host = host.rstrip("/")
        if not om.is_running(host):
            if not om.is_local_host(host):
                self._set("starting", f"Aguardando o servidor Ollama em {host}…")
                for _ in range(self.remote_wait_attempts):  # ex.: docker-compose sobe os serviços em paralelo
                    time.sleep(2)
                    if om.is_running(host):
                        break
                else:
                    self._set("error", f"Não foi possível conectar ao Ollama em {host}. Verifique o endereço nas configurações.")
                    return
        if not om.is_running(host):
            if om.is_local_host(host):
                binary = om.find_binary()
                if not binary:
                    hint = {"linux": "Instale com:  curl -fsSL https://ollama.com/install.sh | sh",
                            "darwin": "Baixe o app em https://ollama.com/download (ou: brew install ollama)"}.get(sys.platform, "")
                    self._set("ollama_missing", "O Ollama é necessário e não foi encontrado neste computador.", detail=hint)
                    return
                self._set("starting", "Iniciando o servidor Ollama…")
                self.process.start(binary)
                for _ in range(60):
                    if om.is_running(host):
                        break
                    time.sleep(0.5)
                else:
                    self._set("error", "O servidor Ollama não respondeu após ser iniciado.")
                    return

        self._set("checking", "Procurando modelos com suporte a áudio…")
        if self.client.audio_models():
            self.mark_ready()
            return

        model = self.settings["base_model"]
        if not self.settings["auto_pull"]:
            self._set("error", f"Nenhum modelo com áudio instalado. Baixe um (ex.: 'ollama pull {model}') ou ative o "
                               "download automático nas configurações.")
            return

        self._set("pulling", f"Baixando o modelo base {model} (pode levar vários minutos)…", 0)

        def on_progress(done, total, status):
            if total:
                self._set(progress=100 * done / total,
                          detail=f"{status} — {done / 1e9:.2f} de {total / 1e9:.2f} GB")
            else:
                self._set(detail=status)

        self.client.pull(model, on_progress)
        if not any(model == m["name"] or m["name"].startswith(model + ":") for m in self.client.audio_models()):
            self._set("error", f"O modelo '{model}' foi baixado mas não declara suporte a áudio. "
                               "Escolha outro modelo base nas configurações.")
            return
        self.mark_ready()

    def _ensure_ffmpeg(self) -> bool:
        """Resolve o FFmpeg (config → PATH → pasta local); se não houver, baixa para a pasta local (Windows)."""
        configured = self.settings["ffmpeg_path"]
        found = fm.resolve(configured)
        if not found and configured:
            self._set("ffmpeg_missing", "O caminho do FFmpeg configurado não contém ffmpeg e ffprobe.",
                      detail="Corrija o caminho em Configurações ou deixe em branco para detecção/instalação automática.")
            return False
        if not found:
            if not fm.can_auto_install():
                self._set("ffmpeg_missing", "O FFmpeg é necessário e não foi encontrado.", detail=FFMPEG_HELP)
                return False
            self._set("installing_ffmpeg", f"FFmpeg não encontrado. Baixando para {fm.local_dir()} …", 0, "")
            try:
                found = fm.download_local(
                    lambda d, t: self._set(progress=100 * d / t, detail=f"{d / 1e6:.0f} de {t / 1e6:.0f} MB"))
            except Exception as e:
                self._set("ffmpeg_missing", f"Não foi possível baixar o FFmpeg: {e}",
                          detail=f"{FFMPEG_HELP}. Depois, informe a pasta em Configurações.")
                return False
        fm.set_active(*found)
        return True

    # --- instalador do Ollama (Windows) -------------------------------
    def install_ollama(self) -> bool:
        if not self.can_auto_install:
            return False
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
            self.state.update(status="installing", message="Baixando o instalador do Ollama…", progress=0.0, detail="")
            self._thread = threading.Thread(target=self._install, daemon=True)
            self._thread.start()
            return True

    def _install(self):
        try:
            path = om.download_installer(
                lambda d, t: self._set(progress=100 * d / t, detail=f"{d / 1e6:.0f} de {t / 1e6:.0f} MB"))
            om.run_installer(path)
            self._set("ollama_missing", "Instalador aberto. Conclua a instalação e clique em “Verificar novamente”.",
                      0, "")
        except Exception as e:
            self._set("error", f"Falha ao baixar/abrir o instalador do Ollama: {e}. "
                               f"Instale manualmente em {om.DOWNLOAD_PAGE}")

    def shutdown(self):
        self.process.stop()
