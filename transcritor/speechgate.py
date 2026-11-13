"""Porteiro de fala (sem LLM e sem PyTorch): decide se um fragmento tem fala ANTES de gastar um modelo de áudio nele.

Por quê: no filme de 45 min o modelo pequeno inventava texto em música e efeitos (inserções de 18,5 % das palavras de referência no
modo 1); o revisor corta parte disso, mas só depois de o gerador já ter inventado. O detector é o VAD Silero em ONNX que já vem com o
faster-whisper (onnxruntime, sem PyTorch). Sem ele o porteiro fica desligado (todos os fragmentos passam).

Calibração (docs/experimento-modo1-v2.md): sinais simples de energia/ritmo (modulação silábica, fluxo espectral, faixa da voz) NÃO
separam música de fala em áudio de filme; o Silero separa: com limiar 0,35, 31 dos 35 trechos sem fala do filme têm ≤ 3 % de fala e
o custo são trechos de fala por cima de música (≈ 2 % das palavras de referência). A regra é conservadora: só descarta quando o Silero
não acha quase nada de fala.
"""
import numpy as np

SAMPLE_RATE = 16000
THRESHOLD = 0.35        # probabilidade mínima do Silero para contar como fala
MIN_SPEECH_MS = 250
MIN_SILENCE_MS = 500
MIN_RATIO = 0.03        # fração do fragmento com fala abaixo da qual o fragmento é descartado

_state: dict = {}


def available() -> bool:
    if "ok" not in _state:
        try:
            from faster_whisper.vad import get_speech_timestamps  # noqa: F401

            _state["ok"] = True
        except Exception:  # faster-whisper/onnxruntime não instalados
            _state["ok"] = False
    return _state["ok"]


def speech_ratio(samples: np.ndarray) -> float:
    """Fração (0–1) do fragmento com fala segundo o Silero; 1.0 quando o detector não está disponível (nada é descartado)."""
    if not available() or len(samples) == 0:
        return 1.0
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    x = np.asarray(samples, dtype=np.float32)
    if float(np.abs(x).max()) > 1.5:  # amostras em escala int16 (o app usa assim)
        x = x / 32768.0
    ts = get_speech_timestamps(x, VadOptions(threshold=THRESHOLD, min_speech_duration_ms=MIN_SPEECH_MS, min_silence_duration_ms=MIN_SILENCE_MS))
    return min(1.0, sum(t["end"] - t["start"] for t in ts) / max(len(x), 1))


def has_speech(samples: np.ndarray) -> bool:
    return speech_ratio(samples) > MIN_RATIO
