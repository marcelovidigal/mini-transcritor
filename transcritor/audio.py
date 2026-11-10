"""Extração de áudio (ffmpeg) e divisão em trechos curtos, preferencialmente em pausas."""
import io
import subprocess
import wave
from dataclasses import dataclass

import numpy as np

from . import ffmpeg_manager
from .envutil import clean_env
from .media import MediaError

SAMPLE_RATE = 16000


@dataclass
class Chunk:
    start: float  # segundos
    end: float
    samples: np.ndarray  # int16 mono


def extract_wav(src: str, dst: str) -> None:
    """Converte qualquer mídia para WAV PCM 16 kHz mono (descarta vídeo)."""
    cmd = [ffmpeg_manager.ffmpeg_path(), "-y", "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", dst]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=clean_env())
    if res.returncode != 0:
        raise MediaError("Falha ao extrair o áudio: " + (res.stderr.strip().splitlines() or ["erro desconhecido"])[-1])


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise MediaError("WAV intermediário em formato inesperado.")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def to_wav_bytes(samples: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(samples.astype(np.int16).tobytes())
    return buf.getvalue()


def _quietest_point(samples: np.ndarray, lo: int, hi: int, frame: int) -> int:
    """Índice do centro do quadro de menor energia em [lo, hi)."""
    n = (hi - lo) // frame
    if n <= 0:
        return hi
    seg = samples[lo : lo + n * frame].astype(np.float64).reshape(n, frame)
    energy = (seg**2).mean(axis=1)
    return lo + int(np.argmin(energy)) * frame + frame // 2


def split_chunks(
    samples: np.ndarray,
    target: float = 20.0,
    maximum: float = 28.0,
    silence_rms: float = 80.0,
) -> list[Chunk]:
    """Divide em trechos de ~`target` s (±4 s), cortando no ponto mais silencioso dessa janela; nunca passa de `maximum`.

    Trechos praticamente silenciosos (RMS < `silence_rms`) são descartados.
    """
    maximum = max(maximum, target)
    total = len(samples)
    tgt, mx = int(target * SAMPLE_RATE), int(maximum * SAMPLE_RATE)
    frame = int(0.05 * SAMPLE_RATE)
    margin = min(int(4 * SAMPLE_RATE), tgt // 2)  # o corte é procurado em [alvo - 4 s, alvo + 4 s]
    chunks: list[Chunk] = []
    pos = 0
    while pos < total:
        if total - pos <= min(tgt + margin, mx):
            end = total
        else:
            lo = pos + tgt - margin
            hi = min(pos + tgt + margin, pos + mx, total)
            end = _quietest_point(samples, lo, hi, frame)
            end = min(max(end, pos + frame), total)
        piece = samples[pos:end]
        if len(piece) and np.sqrt(np.mean(piece.astype(np.float64) ** 2)) >= silence_rms:
            chunks.append(Chunk(pos / SAMPLE_RATE, end / SAMPLE_RATE, piece))
        pos = end
    return chunks
