"""Montagem de cues e formatação SRT."""
import re
from dataclasses import dataclass

MAX_CUE_CHARS = 84  # ~2 linhas de 42 caracteres


@dataclass
class Cue:
    start: float
    end: float
    text: str


def format_timestamp(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")  # tags eventuais
    text = re.sub(r"^\s*(transcri[çc][ãa]o|transcription)\s*:\s*", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip().strip('"“”').strip()


def split_text(text: str, limit: int = MAX_CUE_CHARS) -> list[str]:
    """Quebra por frases e, se necessário, por palavras, respeitando `limit`."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text) if s.strip()]
    parts: list[str] = []
    cur = ""
    for sent in sentences:
        for word in sent.split():
            if cur and len(cur) + 1 + len(word) > limit:
                parts.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}".strip()
        if cur and len(cur) >= limit * 0.5:  # fecha a cue no fim da frase se já está razoável
            parts.append(cur)
            cur = ""
    if cur:
        parts.append(cur)
    return parts


def cues_for_chunk(start: float, end: float, text: str) -> list[Cue]:
    """Distribui o texto de um trecho entre cues, com tempo proporcional ao tamanho."""
    parts = split_text(clean_text(text))
    if not parts:
        return []
    total_chars = sum(len(p) for p in parts)
    dur = max(end - start, 0.001)
    cues, t = [], start
    for i, p in enumerate(parts):
        t_end = end if i == len(parts) - 1 else t + dur * len(p) / total_chars
        cues.append(Cue(t, t_end, p))
        t = t_end
    return cues


def to_srt(cues: list[Cue]) -> str:
    blocks = []
    for i, c in enumerate(cues, 1):
        blocks.append(f"{i}\n{format_timestamp(c.start)} --> {format_timestamp(c.end)}\n{c.text}\n")
    return "\n".join(blocks)
