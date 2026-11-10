"""Classificação de mídia (áudio/vídeo) e sondagem com ffprobe."""
import json
import os
import shutil
import subprocess
from dataclasses import dataclass

from . import ffmpeg_manager
from .envutil import clean_env

VIDEO_EXTS = {".mkv", ".avi", ".mp4", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".ts", ".mpg", ".mpeg", ".3gp", ".ogv"}
AUDIO_EXTS = {".mp3", ".flac", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif", ".amr", ".mka"}


class MediaError(Exception):
    """Erro de domínio relacionado ao arquivo de mídia ou ao ffmpeg."""


@dataclass
class MediaInfo:
    path: str
    kind: str  # "audio" | "video"
    duration: float


def require_ffmpeg() -> None:
    for tool, path in (("ffmpeg", ffmpeg_manager.ffmpeg_path()), ("ffprobe", ffmpeg_manager.ffprobe_path())):
        if shutil.which(path) is None and not os.path.isfile(path):
            raise MediaError(f"'{tool}' não encontrado. Configure o caminho do FFmpeg ou reinicie o preparo do app.")


def kind_from_extension(path: str) -> str | None:
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    return None


def is_media_file(path: str) -> bool:
    return kind_from_extension(path) is not None


def probe(path: str) -> MediaInfo:
    """Valida o arquivo e define se é áudio ou vídeo. Exige ao menos um fluxo de áudio."""
    if not path or not os.path.isfile(path):
        raise MediaError("Arquivo não encontrado. Verifique o caminho informado.")
    if not os.access(path, os.R_OK):
        raise MediaError("Sem permissão de leitura para o arquivo.")
    require_ffmpeg()
    cmd = [ffmpeg_manager.ffprobe_path(), "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=clean_env())
    if res.returncode != 0:
        raise MediaError("Formato de arquivo não reconhecido ou arquivo corrompido.")
    try:
        data = json.loads(res.stdout or "{}")
    except json.JSONDecodeError:
        raise MediaError("Não foi possível ler as informações do arquivo.")
    streams = data.get("streams", [])
    if not any(s.get("codec_type") == "audio" for s in streams):
        raise MediaError("O arquivo não contém nenhuma faixa de áudio.")
    has_video = any(
        s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic") for s in streams
    )
    try:
        duration = float(data.get("format", {}).get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    return MediaInfo(path=path, kind="video" if has_video else "audio", duration=duration)
