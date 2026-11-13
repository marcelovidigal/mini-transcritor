"""Limpeza do texto transcrito, sem usar modelo: remove alucinações típicas e repetições nas emendas dos trechos."""
import re
import unicodedata
from dataclasses import dataclass

# Frases-fantasma que modelos tipo Whisper inventam em trechos sem fala (só descartadas se forem o texto inteiro).
_GHOST = re.compile(
    r"^(legendas?\s+(pela|por|de|feitas? por)\b.*|tradu[çc][ãa]o\s+e\s+legendas?\b.*|amara\.org.*|.*\bamara\.org\b.*"
    r"|obrigad[oa]( por assistir| pela aten[çc][ãa]o)?|inscreva-?se( no canal)?|"
    r"thanks? for watching|thank you for watching|subtitles? by\b.*|subtitles? made by\b.*|please subscribe|"
    r"like and subscribe|www\.\S+)$",
    re.I,
)
MAX_PLAUSIBLE_CPS = 45.0  # fala humana muito rápida passa pouco de ~25 caracteres/s
MIN_REPEATS = 4  # um mesmo trecho de palavras repetido tantas vezes seguidas é laço do modelo
MIN_SEAM_WORDS = 4


def _norm_words(text: str) -> list[str]:
    t = unicodedata.normalize("NFD", text.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"[^\w\s]", " ", t).split()


def collapse_repeats(text: str) -> str:
    """'e e e e e e' → 'e'; 'não sei não sei não sei não sei' → 'não sei' (4 ou mais repetições seguidas)."""
    words = text.split()
    norm = [re.sub(r"[^\w]", "", w.lower()) for w in words]
    for n in range(1, 7):
        i = 0
        while i + n * MIN_REPEATS <= len(words):
            block = norm[i : i + n]
            reps = 1
            while norm[i + reps * n : i + (reps + 1) * n] == block:
                reps += 1
            if reps >= MIN_REPEATS and any(block):
                words[i + n : i + reps * n] = []
                norm[i + n : i + reps * n] = []
            i += 1
    return " ".join(words)


def is_ghost(text: str) -> bool:
    return bool(_GHOST.match(re.sub(r"\s+", " ", text).strip(" .!?…\"'“”")))


def strip_seam_repeat(prev: str, text: str) -> str:
    """Se o texto abre repetindo o fim do trecho anterior (≥4 palavras), remove a repetição."""
    a, b = _norm_words(prev), _norm_words(text)
    for k in range(min(len(a), len(b), 12), MIN_SEAM_WORDS - 1, -1):
        if a[-k:] == b[:k]:
            return " ".join(text.split()[k:]) if len(text.split()) > k else ""
    return text


@dataclass
class CleanReport:
    ghosts: int = 0
    too_fast: int = 0
    collapsed: int = 0
    seams: int = 0

    def messages(self) -> list[str]:
        out = []
        if self.ghosts:
            out.append(f"{self.ghosts} trecho(s) descartado(s) por parecerem frases inventadas pelo modelo (ex.: \"Legendas pela comunidade…\").")
        if self.too_fast:
            out.append(f"{self.too_fast} trecho(s) descartado(s): texto longo demais para a duração do áudio (provável alucinação).")
        if self.collapsed:
            out.append(f"{self.collapsed} repetição(ões) em laço removida(s).")
        if self.seams:
            out.append(f"{self.seams} repetição(ões) na emenda entre trechos removida(s).")
        return out


def clean_texts(texts: list[str], durations: list[float]) -> tuple[list[str], CleanReport]:
    """Aplica, na ordem: frase-fantasma, velocidade implausível, laços de repetição e repetição na emenda."""
    rep = CleanReport()
    out: list[str] = []
    prev = ""
    for text, dur in zip(texts, durations):
        t = text
        if t:
            if is_ghost(t):
                rep.ghosts += 1
                t = ""
            elif len(t) / max(dur, 0.5) > MAX_PLAUSIBLE_CPS:
                rep.too_fast += 1
                t = ""
        if t:
            c = collapse_repeats(t)
            if c != t:
                rep.collapsed += 1
                t = c
        if t and prev:
            s = strip_seam_repeat(prev, t)
            if s != t:
                rep.seams += 1
                t = s
        out.append(t)
        if t:
            prev = t
    return out, rep
