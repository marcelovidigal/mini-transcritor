"""Montagem de cues e formatação SRT."""
import re
from dataclasses import dataclass

# Regras de legenda (Netflix Timed Text Style Guide — Portuguese (EMEA), conferido em 2026-11-10).
# VERIFICADAS na página: 42 caracteres por linha, no máximo 2 linhas, texto em 1 linha quando couber,
# preferência por pirâmide com a linha de baixo maior (sem deixar 1–2 palavras na de cima), até 17 caracteres/s
# (adultos). NÃO constam da página (valores usuais da indústria, ajustáveis): duração mínima/máxima, intervalo
# mínimo entre legendas e extensão do tempo final.
MAX_LINE_CHARS = 42
MAX_LINES = 2
MAX_CUE_CHARS = MAX_LINE_CHARS * MAX_LINES  # 84
MAX_CPS = 17.0
MIN_DURATION = 5 / 6  # 20 quadros a 24 fps
MAX_DURATION = 7.0
MIN_GAP = 2 / 24  # 2 quadros a 24 fps
MAX_EXTEND = 0.5  # quanto o tempo final pode passar do fim da fala para dar tempo de leitura


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


def _wrappable(text: str, width: int = MAX_LINE_CHARS) -> bool:
    """True se o texto cabe em 1 linha ou em 2 linhas de até `width` caracteres (alguma quebra de palavra serve)."""
    text = " ".join(text.split())
    if len(text) <= width:
        return True
    words = text.split()
    return any(len(" ".join(words[:i])) <= width and len(" ".join(words[i:])) <= width for i in range(1, len(words)))


def _balanced(sentence: str, limit: int) -> list[str]:
    """Divide uma frase longa em partes de tamanho parecido (não 84 + um resto curto), que cabem em 2 linhas,
    preferindo cortar após vírgula e evitando terminar em artigo/preposição."""
    words = sentence.split()
    out: list[str] = []
    while words:
        rest = " ".join(words)
        if len(rest) <= limit and _wrappable(rest):
            out.append(rest)
            break
        target = len(rest) / max(2, -(-len(rest) // limit))
        cands = []
        for j in range(1, len(words)):
            part = " ".join(words[:j])
            if len(part) > limit:
                break
            if _wrappable(part):
                cands.append(j)

        def score(j: int) -> float:
            last = words[j - 1]
            sc = abs(len(" ".join(words[:j])) - target)
            if last[-1] in ",;:":
                sc -= 8
            elif last.lower() in _NO_END:
                sc += 10
            return sc

        j = min(cands, key=score) if cands else 1
        out.append(" ".join(words[:j]))
        words = words[j:]
    return out


def split_text(text: str, limit: int = MAX_CUE_CHARS) -> list[str]:
    """Quebra por frases em partes que cabem em 2 linhas de 42 caracteres. Frases curtas seguidas são agrupadas;
    uma frase maior que `limit` é dividida em partes equilibradas (`_balanced`)."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text) if s.strip()]
    parts: list[str] = []
    cur = ""
    for sent in sentences:
        if len(sent) > limit or not _wrappable(sent):
            if cur:
                parts.append(cur)
                cur = ""
            parts.extend(_balanced(sent, limit))
            continue
        if cur and len(cur) + 1 + len(sent) <= limit and _wrappable(f"{cur} {sent}"):
            cur = f"{cur} {sent}"
        else:
            if cur:
                parts.append(cur)
            cur = sent
        if len(cur) >= limit * 0.5:  # fecha a cue no fim da frase se já está razoável
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


# palavras que não devem terminar a linha de cima (artigos, preposições, conjunções, pronomes: pt e en)
_NO_END = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "por", "pelo", "pela", "para", "pra", "com", "sem", "e", "ou", "que", "se", "mas", "ao", "aos", "à", "às",
    "the", "an", "of", "to", "in", "on", "at", "for", "and", "or", "but", "with", "from", "by", "that", "is",
}


def wrap_lines(text: str, width: int = MAX_LINE_CHARS) -> str:
    """Uma linha se couber; senão duas, em pirâmide (linha de baixo maior), sem deixar 1–2 palavras em cima
    nem terminar a linha de cima em artigo/preposição, preferindo cortar após pontuação."""
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    words = text.split()
    best = None
    for i in range(1, len(words)):
        l1, l2 = " ".join(words[:i]), " ".join(words[i:])
        if len(l1) > width or len(l2) > width:
            continue
        score = abs(len(l1) - 0.8 * len(l2))
        last = words[i - 1]
        if i < 3:
            score += 10  # evita 1–2 palavras na linha de cima
        if last[-1] in ".!?…":
            score -= 8
        elif last[-1] in ",;:":
            score -= 5
        elif last.lower() in _NO_END:
            score += 12
        if best is None or score < best[0]:
            best = (score, l1, l2)
    if best:
        return best[1] + "\n" + best[2]
    cut = text.rfind(" ", 0, width + 1)  # texto maior que 2 linhas: corta no limite
    return text if cut <= 0 else text[:cut] + "\n" + text[cut + 1 :]


def _chars(text: str) -> int:
    return len(text.replace("\n", ""))


def apply_style(cues: list[Cue]) -> list[Cue]:
    """Aplica as regras de legenda: quebra em linhas, duração máxima, intervalo mínimo entre legendas e,
    quando há folga, extensão do tempo final até a velocidade de leitura/duração mínima."""
    out = [Cue(c.start, c.end, wrap_lines(c.text)) for c in cues]
    for i, c in enumerate(out):
        nxt = out[i + 1].start if i + 1 < len(out) else None
        if c.end - c.start > MAX_DURATION:
            c.end = c.start + MAX_DURATION
        if nxt is not None and c.end > nxt - MIN_GAP:
            c.end = max(nxt - MIN_GAP, c.start + 0.2)
        want = max(MIN_DURATION, _chars(c.text) / MAX_CPS)
        if c.end - c.start < want:
            room = (nxt - MIN_GAP - c.end) if nxt is not None else MAX_EXTEND
            c.end += max(0.0, min(want - (c.end - c.start), MAX_EXTEND, room))
    return out


def fast_reading(cues: list[Cue]) -> int:
    """Quantas legendas passam do limite de caracteres por segundo."""
    return sum(1 for c in cues if _chars(c.text) / max(c.end - c.start, 0.001) > MAX_CPS + 0.01)


def to_srt(cues: list[Cue]) -> str:
    blocks = []
    for i, c in enumerate(cues, 1):
        blocks.append(f"{i}\n{format_timestamp(c.start)} --> {format_timestamp(c.end)}\n{c.text}\n")
    return "\n".join(blocks)
