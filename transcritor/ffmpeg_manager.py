"""Localiza (config → PATH → pasta local do app) e instala localmente o FFmpeg/FFprobe."""
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

import requests

from .envutil import clean_env

# Build "essentials" do gyan.dev (ffmpeg + ffprobe, ~80 MB compactado). Só há download automático no Windows.
WINDOWS_ZIP_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
DOWNLOAD_PAGE = "https://ffmpeg.org/download.html"
_EXE = ".exe" if sys.platform == "win32" else ""

_active = {"ffmpeg": None, "ffprobe": None}


def app_dir() -> str:
    """Pasta do app: ao lado do executável (PyInstaller) ou raiz do projeto (código-fonte)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def local_dir() -> str:
    return os.environ.get("MINITRANSCRITOR_FFMPEG_DIR") or os.path.join(app_dir(), "ffmpeg")


def can_auto_install() -> bool:
    return sys.platform == "win32"


def _find_in_dir(directory: str) -> tuple[str, str] | None:
    for d in (directory, os.path.join(directory, "bin")):
        ff, fp = os.path.join(d, "ffmpeg" + _EXE), os.path.join(d, "ffprobe" + _EXE)
        if os.path.isfile(ff) and os.path.isfile(fp):
            return ff, fp
    return None


def find_in_path_or_dir(path: str) -> tuple[str, str] | None:
    """`path` pode ser a pasta das ferramentas, a raiz do FFmpeg (com bin/) ou o próprio ffmpeg(.exe)."""
    path = os.path.expandvars(os.path.expanduser(path.strip().strip('"')))
    if os.path.isfile(path):
        path = os.path.dirname(path)
    return _find_in_dir(path) if os.path.isdir(path) else None


def resolve(configured: str = "") -> tuple[str, str] | None:
    """Ordem: caminho configurado → PATH do sistema → pasta local do app. None se nenhum servir."""
    if configured and configured.strip():
        return find_in_path_or_dir(configured)  # caminho explícito inválido não cai em outro silenciosamente
    ff, fp = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if ff and fp:
        return ff, fp
    return _find_in_dir(local_dir())


def set_active(ffmpeg: str | None, ffprobe: str | None) -> None:
    _active["ffmpeg"], _active["ffprobe"] = ffmpeg, ffprobe


def ffmpeg_path() -> str:
    return _active["ffmpeg"] or shutil.which("ffmpeg") or "ffmpeg"


def ffprobe_path() -> str:
    return _active["ffprobe"] or shutil.which("ffprobe") or "ffprobe"


def active_description() -> str:
    return _active["ffmpeg"] or ""


def verify(ffmpeg: str) -> bool:
    try:
        return subprocess.run([ffmpeg, "-version"], capture_output=True, timeout=30, env=clean_env()).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def download_local(progress=None, url: str = WINDOWS_ZIP_URL) -> tuple[str, str]:
    """Baixa o FFmpeg e extrai ffmpeg/ffprobe para a pasta local do app. Devolve os caminhos."""
    dest = local_dir()
    os.makedirs(dest, exist_ok=True)
    fd, tmp_zip = tempfile.mkstemp(suffix=".zip", prefix="ffmpeg_")
    os.close(fd)
    try:
        with requests.get(url, stream=True, timeout=(10, 60)) as r:
            r.raise_for_status()
            total, done = int(r.headers.get("content-length") or 0), 0
            with open(tmp_zip, "wb") as f:
                for block in r.iter_content(1024 * 256):
                    f.write(block)
                    done += len(block)
                    if progress and total:
                        progress(done, total)
        wanted = {"ffmpeg" + _EXE, "ffprobe" + _EXE}
        with zipfile.ZipFile(tmp_zip) as z:
            for member in z.namelist():
                name = os.path.basename(member)
                if name in wanted and "/bin/" in "/" + member.replace("\\", "/"):
                    with z.open(member) as src, open(os.path.join(dest, name), "wb") as out:
                        shutil.copyfileobj(src, out)
                    wanted.discard(name)
        if wanted:
            raise RuntimeError("O pacote baixado não contém ffmpeg/ffprobe.")
    finally:
        try:
            os.remove(tmp_zip)
        except OSError:
            pass
    found = _find_in_dir(dest)
    if not found or not verify(found[0]):
        raise RuntimeError("O FFmpeg baixado não pôde ser executado.")
    return found
