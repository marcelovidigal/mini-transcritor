"""Especialistas da opção Qualidade: ajustes finais de texto ANTES de a legenda ser montada. Cada um tem travas em código,
então um modelo que "reescreve demais" não consegue estragar o resultado (a mudança é descartada).

  1. Termos      — código puro: aplica a grafia de nomes/termos (glossário do usuário + termos ouvidos pelo revisor) em palavras
                   quase iguais ("Odiseu" → "Odysseus"). Só troca palavras muito parecidas (difflib ≥ 0,84).
  2. Pontuação   — texto no Ollama: pontuação, maiúsculas e acentos; as PALAVRAS não podem mudar (senão é descartado).
  3. Tradução    — texto no Ollama, depois da tradução: confere cada (original, tradução); trava de números, tamanho e vazio.

Falha do modelo (OllamaError, formato inesperado) nunca interrompe o trabalho: o texto anterior é mantido.
"""
import difflib
import re

from . import metrics
from .ollama_client import OllamaError

SNAP_RATIO = 0.84
BATCH = 8
_TOKEN = re.compile(r"^(\W*)(.*?)(\W*)$", re.S)


def build_glossary(context: str, reviews: list) -> list[str]:
    """Termos a respeitar: os do contexto do usuário (separados por vírgula/linha) + os que o revisor disse ter ouvido."""
    found: list[str] = []
    seen: set[str] = set()

    def add(t):
        t = " ".join(str(t).split()).strip(" .,;:!?\"'")
        if len(t) >= 3 and t.casefold() not in seen:
            seen.add(t.casefold())
            found.append(t)

    for part in re.split(r"[,;\n]", context or ""):
        if len(part.split()) <= 4:
            add(part)
    for r in reviews:
        for t in (r or {}).get("terms") or []:
            if isinstance(t, str):
                add(t)
    return found[:80]


def snap_terms(text: str, terms: list[str]) -> str:
    """Troca palavras (ou sequências de palavras) muito parecidas com um termo do glossário pela grafia do glossário."""
    if not text or not terms:
        return text
    toks = text.split(" ")
    out, i = [], 0
    by_len = sorted(terms, key=lambda t: -len(t.split()))
    while i < len(toks):
        hit = None
        for term in by_len:
            n = len(term.split())
            window = toks[i:i + n]
            if len(window) < n:
                continue
            core = " ".join(_TOKEN.match(w).group(2) for w in window)
            if len(core) < 4 or core == term or core[0].casefold() != term[0].casefold():
                continue
            if difflib.SequenceMatcher(None, core.casefold(), term.casefold()).ratio() >= SNAP_RATIO:
                lead, trail = _TOKEN.match(window[0]).group(1), _TOKEN.match(window[-1]).group(3)
                hit = (n, lead + term + trail)
                break
        if hit:
            out.append(hit[1])
            i += hit[0]
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


def apply_terms(texts: list[str], terms: list[str]) -> tuple[list[str], int]:
    new = [snap_terms(t, terms) for t in texts]
    return new, sum(1 for a, b in zip(texts, new) if a != b)


def _batches(idx: list[int], size: int) -> list[list[int]]:
    return [idx[k:k + size] for k in range(0, len(idx), size)]


def polish(client, model: str, texts: list[str], language: str, glossary: str = "", *, check=None) -> tuple[list[str], dict]:
    """Especialista de pontuação. Aceita a versão do modelo só se as palavras forem exatamente as mesmas."""
    lang = (language or "").lower().split("-")[0]
    out = list(texts)
    stats = {"changed": 0, "rejected": 0, "failed_batches": 0}
    for batch in _batches([i for i, t in enumerate(texts) if t], BATCH):
        if check:
            check()
        try:
            res = client.polish(model, [texts[i] for i in batch], language, glossary)
        except OllamaError:
            stats["failed_batches"] += 1
            continue
        for i, new in zip(batch, res):
            new = " ".join(str(new).split())
            if not new or new == texts[i]:
                continue
            if metrics.normalize(new, lang) == metrics.normalize(texts[i], lang) and len(new) <= len(texts[i]) * 1.3 + 4:
                out[i] = new
                stats["changed"] += 1
            else:
                stats["rejected"] += 1
    return out, stats


def _digits(t: str) -> list[str]:
    return sorted(re.findall(r"\d+", t))


def review_translation(client, model: str, pairs: list[tuple[str, str]], language: str, glossary: str = "", *, check=None) -> tuple[list[str], dict]:
    """Especialista de tradução. Trava: não vazio, mesmos números, tamanho entre 60 % e 150 % da tradução anterior."""
    out = [t for _s, t in pairs]
    stats = {"changed": 0, "rejected": 0, "failed_batches": 0}
    for batch in _batches([i for i, (s, t) in enumerate(pairs) if s and t], BATCH):
        if check:
            check()
        try:
            res = client.review_translation(model, [pairs[i] for i in batch], language, glossary)
        except OllamaError:
            stats["failed_batches"] += 1
            continue
        for i, new in zip(batch, res):
            old, new = pairs[i][1], " ".join(str(new).split())
            if not new or new == old:
                continue
            if _digits(new) == _digits(old) and 0.6 * len(old) <= len(new) <= 1.5 * len(old) + 4:
                out[i] = new
                stats["changed"] += 1
            else:
                stats["rejected"] += 1
    return out, stats
