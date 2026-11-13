"""Juiz da v2 do motor Qualidade (docs/plano-pipeline-v2.md): o WhisperX (B) é a âncora e o A é um desafiante.

Mudanças em relação à v1 (quality_engine.judge), todas decididas pelos dados do experimento do filme:
  * **Silêncio é informação:** quando o B calou e só o A escreveu, o padrão é "sem fala" (no filme, 19 de 37 desses trechos eram
    música/efeito, e o A inventou palavras). O A só vence se o revisor OUVIR fala e o A for confiável neste arquivo.
  * **Confiança do A medida no arquivo, sem referência:** razão de palavras A/B e fração de fragmentos "só A" (clipes de fala limpa:
    0,98–1,06 e 0 %; filme: 1,19–1,23 e 19–22 %). Com o A pouco confiável, ele não muda o texto do B.
  * **Decisão por ponto de divergência:** o código alinha A e B palavra a palavra; o revisor escolhe A ou B em cada ponto e o texto final é
    MONTADO pelo código (base B + trechos escolhidos de A). Nada de texto livre do revisor.
  * **Correções do B pelo áudio:** o revisor pode corrigir uma palavra que os DOIS erraram (`fixes`), só se a nova palavra se parecer com
    a antiga (difflib) ou vier do glossário: é o único caminho para passar do B quando o A não acrescenta nada.
"""
import difflib
import re

from . import metrics

MAX_DIFFS = 10  # acima disto o fragmento diverge demais para decisão ponto a ponto: vale o B (âncora)
RATIO_LOW = 1.10  # A escrevendo mais de 10 % de palavras que o B = excesso (alucinação/paráfrase)
ONLY_A_LOW = 0.05  # mais de 5 % dos fragmentos com texto só no A = o A preenche silêncio do B
FIX_SIMILARITY = 0.6
_KEY = re.compile(r"\w+", re.UNICODE)


def _key(tok: str) -> str:
    return "".join(_KEY.findall(tok)).casefold()


def _ops(a: str, b: str) -> tuple[list[str], list[str], list[tuple]]:
    at, bt = a.split(), b.split()
    sm = difflib.SequenceMatcher(None, [_key(t) for t in at], [_key(t) for t in bt], autojunk=False)
    return at, bt, sm.get_opcodes()


def word_diffs(a: str, b: str) -> list[dict]:
    """Pontos de divergência entre A e B: [{"a": trecho do A, "b": trecho do B}] (um dos lados pode ser vazio: inserção/omissão)."""
    at, bt, ops = _ops(a, b)
    return [{"a": " ".join(at[i1:i2]), "b": " ".join(bt[j1:j2])} for tag, i1, i2, j1, j2 in ops if tag != "equal"]


def compose(a: str, b: str, picks: list[str]) -> str:
    """Texto final: base B; em cada ponto de divergência vale o lado escolhido (`picks[k]` = "A" ou "B")."""
    at, bt, ops = _ops(a, b)
    out, k = [], 0
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            out += bt[j1:j2]
        else:
            out += at[i1:i2] if str(picks[k]).strip().upper() == "A" else bt[j1:j2]
            k += 1
    return " ".join(out)


def file_signals(a_texts: list[str], b_texts: list[str], language: str = "") -> dict:
    """Sinais (sem referência) de que o A é confiável neste arquivo."""
    lang = (language or "").lower().split("-")[0]
    wa = sum(len(metrics.normalize(t, lang)) for t in a_texts)
    wb = sum(len(metrics.normalize(t, lang)) for t in b_texts)
    n = max(1, len(a_texts))
    only_a = sum(1 for x, y in zip(a_texts, b_texts) if x and not y)
    return {"ratio_a_b": round(wa / wb, 3) if wb else (float("inf") if wa else 1.0), "only_a_rate": round(only_a / n, 3),
            "words_a": wa, "words_b": wb, "only_a": only_a}


def a_trusted(signals: dict) -> bool:
    return signals.get("ratio_a_b", 1.0) <= RATIO_LOW and signals.get("only_a_rate", 0.0) <= ONLY_A_LOW


def apply_fixes(text: str, fixes, glossary_terms=()) -> tuple[str, int]:
    """Troca palavras do `text` segundo as correções do revisor ({"from","to"}); só aceita se `to` se parece com `from` ou está no glossário."""
    if not text or not isinstance(fixes, list):
        return text, 0
    gloss = {_key(t): t for t in glossary_terms}
    toks, done = text.split(), 0
    for f in fixes:
        if not isinstance(f, dict):
            continue
        src, dst = str(f.get("from") or "").strip(), str(f.get("to") or "").strip()
        if not src or not dst or " " in src or " " in dst or _key(src) == _key(dst):
            continue
        ok = _key(dst) in gloss or difflib.SequenceMatcher(None, _key(src), _key(dst)).ratio() >= FIX_SIMILARITY
        if not ok:
            continue
        for i, t in enumerate(toks):
            if _key(t) == _key(src):
                m = re.match(r"^(\W*)(.*?)(\W*)$", t, re.S)
                toks[i] = m.group(1) + dst + m.group(3)
                done += 1
                break
    return " ".join(toks), done


def decide(a: str, b: str, review: dict | None, trusted: bool, glossary_terms=(), min_confidence: float = 0.5) -> tuple[str, str]:
    """(texto final, origem). Origens: B | B+correção | A+B (pontos escolhidos) | A(fala confirmada) | silêncio | A."""
    if not a and not b:
        return "", "B"
    r = review or {}
    try:
        conf = float(r.get("confidence") or 0)
    except (TypeError, ValueError):
        conf = 0.0
    sure = bool(r) and conf >= min_confidence
    if not b:  # só o A escreveu: o B ouviu silêncio
        if sure and r.get("speech") is True and trusted:
            return a, "A(fala confirmada)"
        return "", "silêncio"
    if not a:
        text, n = apply_fixes(b, r.get("fixes") if sure else None, glossary_terms)
        return text, "B+correção" if n else "B"
    diffs = word_diffs(a, b)
    text, src = b, "B"
    picks = r.get("picks") if sure else None
    if diffs and trusted and isinstance(picks, list) and len(picks) == len(diffs) and len(diffs) <= MAX_DIFFS:
        if any(str(p).strip().upper() == "A" for p in picks):
            text, src = compose(a, b, picks), "A+B"
    text, n = apply_fixes(text, r.get("fixes") if sure else None, glossary_terms)
    if n:
        src += "+correção"
    return text, src
