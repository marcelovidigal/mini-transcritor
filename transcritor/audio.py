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


def detect_speech(
    samples: np.ndarray,
    min_rms: float = 80.0,
    frame_ms: int = 30,
    min_silence: float = 0.35,
    min_speech: float = 0.2,
    pad: float = 0.3,
) -> list[tuple[int, int]]:
    """VAD por energia (sem dependências): devolve trechos de fala como (início, fim) em amostras.

    O limiar se adapta ao áudio (10% do nível de fala, nunca abaixo de `min_rms`). Pausas menores que `min_silence`
    são fechadas, falas menores que `min_speech` descartadas e cada fala ganha `pad` s de folga nas bordas (0,3 s: com 0,1 s o início/fim das palavras era cortado; medido em 2026-11-11).
    """
    total = len(samples)
    frame = int(SAMPLE_RATE * frame_ms / 1000)
    n = total // frame
    if n == 0:
        return []
    seg = samples[: n * frame].astype(np.float32).reshape(n, frame)
    rms = np.sqrt((seg**2).mean(axis=1))
    thr = max(min_rms, 0.1 * float(np.percentile(rms, 95)))
    v = np.concatenate(([0], (rms >= thr).astype(np.int8), [0]))
    d = np.diff(v)
    runs = [[int(s), int(e)] for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
    merged: list[list[int]] = []
    gap_frames = min_silence * 1000 / frame_ms
    for s, e in runs:
        if merged and s - merged[-1][1] < gap_frames:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    min_frames = min_speech * 1000 / frame_ms
    spans = [(s * frame, e * frame) for s, e in merged if e - s >= min_frames]
    padn = int(pad * SAMPLE_RATE)
    out: list[tuple[int, int]] = []
    for i, (s, e) in enumerate(spans):
        prev_e = spans[i - 1][1] if i else 0
        next_s = spans[i + 1][0] if i + 1 < len(spans) else total
        out.append((s - min(padn, (s - prev_e) // 2 if i else s), e + min(padn, (next_s - e) // 2 if i + 1 < len(spans) else total - e)))
    return out


def split_by_speech(samples: np.ndarray, target: float = 12.0, maximum: float = 28.0) -> list[Chunk]:
    """Divide o áudio em trechos de fala: pula o silêncio e só corta nas pausas.

    Agrupa falas consecutivas até chegar a ~`target` s (o corte é a pausa seguinte); nunca passa de `maximum`.
    Uma fala contínua maior que `maximum` (sem pausas) é dividida no ponto mais silencioso (`split_chunks`).
    """
    maximum = max(maximum, target)
    tgt, mx = int(target * SAMPLE_RATE), int(maximum * SAMPLE_RATE)
    chunks: list[Chunk] = []

    def emit(start: int, end: int):
        if end - start <= mx:
            chunks.append(Chunk(start / SAMPLE_RATE, end / SAMPLE_RATE, samples[start:end]))
            return
        for c in split_chunks(samples[start:end], target, maximum, silence_rms=0):
            off = start / SAMPLE_RATE
            chunks.append(Chunk(c.start + off, c.end + off, c.samples))

    cur_start = cur_end = None
    for s, e in detect_speech(samples):
        if cur_start is None:
            cur_start, cur_end = s, e
        elif e - cur_start > mx:  # juntar estouraria o limite: fecha o trecho anterior na pausa
            emit(cur_start, cur_end)
            cur_start, cur_end = s, e
        else:
            cur_end = e
        if cur_end - cur_start >= tgt:  # já tem o tamanho-alvo: corta na pausa que vem a seguir
            emit(cur_start, cur_end)
            cur_start = cur_end = None
    if cur_start is not None:
        emit(cur_start, cur_end)
    return chunks


def normalize_peak(samples: np.ndarray, peak: float = 29000.0, max_gain: float = 8.0) -> np.ndarray:
    """Ganho para o pico chegar perto do máximo (escala int16), no máximo `max_gain`x: mesma fala, entrada diferente para o modelo."""
    top = float(np.abs(samples).max()) if len(samples) else 0.0
    if top <= 0:
        return samples
    return np.clip(samples.astype(np.float32) * min(max_gain, peak / top), -32768, 32767).astype(samples.dtype)


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
