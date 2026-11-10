"""Localizar, iniciar e parar o Ollama local; baixar o instalador oficial."""
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.parse

import requests

from .envutil import clean_env

DOWNLOAD_PAGE = "https://ollama.com/download"
WINDOWS_INSTALLER_URL = "https://ollama.com/download/OllamaSetup.exe"


def is_local_host(host: str) -> bool:
    name = (urllib.parse.urlparse(host).hostname or "").lower()
    return name in ("127.0.0.1", "localhost", "::1", "0.0.0.0")


def find_binary() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    candidates = []
    if sys.platform == "win32":
        candidates.append(os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe"))
        candidates.append(os.path.join(os.environ.get("ProgramFiles", ""), "Ollama", "ollama.exe"))
    elif sys.platform == "darwin":
        candidates += ["/usr/local/bin/ollama", "/opt/homebrew/bin/ollama",
                       "/Applications/Ollama.app/Contents/Resources/ollama"]
    else:
        candidates += ["/usr/local/bin/ollama", "/usr/bin/ollama"]
    return next((c for c in candidates if c and os.path.isfile(c)), None)


def is_running(host: str, timeout: float = 2) -> bool:
    try:
        return requests.get(host.rstrip("/") + "/api/version", timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


class OllamaProcess:
    """Processo `ollama serve` iniciado por este app (e só ele é encerrado na saída)."""

    def __init__(self):
        self._proc: subprocess.Popen | None = None

    def start(self, binary: str) -> None:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self._proc = subprocess.Popen(
            [binary, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, creationflags=flags, env=clean_env(),
        )

    @property
    def started_by_us(self) -> bool:
        return self._proc is not None

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None


def download_installer(progress=None) -> str:
    """Baixa o instalador oficial (Windows) para a pasta temporária e devolve o caminho."""
    dest_dir = tempfile.mkdtemp(prefix="ollama_setup_")
    dest = os.path.join(dest_dir, "OllamaSetup.exe")
    with requests.get(WINDOWS_INSTALLER_URL, stream=True, timeout=(10, 60)) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        with open(dest, "wb") as f:
            for block in r.iter_content(1024 * 256):
                f.write(block)
                done += len(block)
                if progress and total:
                    progress(done, total)
    return dest


def run_installer(path: str) -> None:
    os.startfile(path)  # noqa: S606 - abre o instalador oficial com a interface padrão do Windows
