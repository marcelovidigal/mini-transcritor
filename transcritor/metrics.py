"""Métricas de avaliação de legendas: WER, regras de legenda e erro de tempo (usadas por tools/evaluate.py)."""
import re
import statistics
import unicodedata

from . import srt
from .srt import Cue


_PT_UNITS = ["zero", "um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito", "nove", "dez", "onze", "doze", "treze",
             "catorze", "quinze", "dezesseis", "dezessete", "dezoito", "dezenove"]
_PT_TENS = ["", "", "vinte", "trinta", "quarenta", "cinquenta", "sessenta", "setenta", "oitenta", "noventa"]
_PT_HUND = ["", "cento", "duzentos", "trezentos", "quatrocentos", "quinhentos", "seiscentos", "setecentos", "oitocentos", "novecentos"]
_EN_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
             "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _spell_pt(n: int) -> str:
    if n < 20:
        return _PT_UNITS[n]
    if n < 100:
        return _PT_TENS[n // 10] + (f" e {_PT_UNITS[n % 10]}" if n % 10 else "")
    if n == 100:
        return "cem"
    if n < 1000:
        return _PT_HUND[n // 100] + (f" e {_spell_pt(n % 100)}" if n % 100 else "")
    return str(n)


def _spell_en(n: int) -> str:
    if n < 20:
        return _EN_UNITS[n]
    if n < 100:
        return _EN_TENS[n // 10] + (f" {_EN_UNITS[n % 10]}" if n % 10 else "")
    if n < 1000:
        return _EN_UNITS[n // 100] + " hundred" + (f" {_spell_en(n % 100)}" if n % 100 else "")
    return str(n)


def spell_numbers(text: str, lang: str | None) -> str:
    """Escreve por extenso os inteiros de 0 a 999 (e '%') em pt ou en, para o WER não punir '25' contra 'vinte e cinco'.
    Outros idiomas ou números maiores ficam como estão."""
    base = (lang or "").lower().split("-")[0]
    if base not in ("pt", "en"):
        return text
    spell = _spell_pt if base == "pt" else _spell_en
    percent = " por cento" if base == "pt" else " percent"
    text = re.sub(r"(\d+)\s*%", lambda m: m.group(1) + percent, text)
    return re.sub(r"\b\d{1,3}\b", lambda m: spell(int(m.group(0))), text)


def normalize(text: str, lang: str | None = None) -> list[str]:
    """Minúsculas, sem acentos nem pontuação, em palavras. Com `lang` (pt/en), números viram palavras (spell_numbers)."""
    t = unicodedata.normalize("NFD", spell_numbers(text or "", lang).lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"[^\w\s]", " ", t).split()


def wer(reference: str, hypothesis: str, lang: str | None = None) -> float:
    """Taxa de erro de palavras (0 = perfeito; pode passar de 1). Texto de referência vazio → 0 ou 1."""
    r, h = normalize(reference, lang), normalize(hypothesis, lang)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)


_TS = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def parse_srt(text: str) -> list[Cue]:
    cues = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()):
        lines = block.split("\n")
        for k, line in enumerate(lines):
            m = _TS.search(line)
            if m:
                g = [int(x) for x in m.groups()]
                start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
                end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
                cues.append(Cue(start, end, "\n".join(lines[k + 1 :]).strip()))
                break
    return cues


def plain_text(cues: list[Cue]) -> str:
    return " ".join(c.text.replace("\n", " ") for c in cues)


def rule_stats(cues: list[Cue]) -> dict:
    """Quantas legendas violam cada regra (ver srt.py) — para acompanhar a qualidade formal da saída."""
    n = len(cues)
    over_cps = over_chars = over_lines = short = long_ = overlap = tight_gap = 0
    for i, c in enumerate(cues):
        ls = c.text.split("\n")
        dur = c.end - c.start
        over_lines += len(ls) > srt.MAX_LINES
        over_chars += any(len(x) > srt.MAX_LINE_CHARS for x in ls)
        over_cps += srt._chars(c.text) / max(dur, 0.001) > srt.MAX_CPS + 0.01
        short += dur < srt.MIN_DURATION - 1e-6
        long_ += dur > srt.MAX_DURATION + 1e-6
        if i + 1 < n:
            gap = cues[i + 1].start - c.end
            overlap += gap < -1e-6
            tight_gap += -1e-6 <= gap < srt.MIN_GAP - 0.0015  # tolera o arredondamento do SRT (milissegundos)
    return {"cues": n, "over_cps": over_cps, "line_too_long": over_chars, "too_many_lines": over_lines,
            "too_short": short, "too_long": long_, "overlap": overlap, "gap_too_small": tight_gap}


def timing_error(reference: list[Cue], hypothesis: list[Cue], window: float = 10.0) -> dict:
    """Para cada legenda de referência, acha a da hipótese com mais palavras em comum (±`window` s) e mede
    a diferença entre os inícios. Devolve mediana, p90 e a fração de referências sem par."""
    errs, unmatched = [], 0
    hyp_words = [set(normalize(h.text)) for h in hypothesis]
    for r in reference:
        rw = set(normalize(r.text))
        best, best_score = None, 0.0
        for h, hw in zip(hypothesis, hyp_words):
            if abs(h.start - r.start) > window or not rw or not hw:
                continue
            score = len(rw & hw) / len(rw | hw)
            if score > best_score:
                best, best_score = h, score
        if best is None or best_score < 0.3:
            unmatched += 1
        else:
            errs.append(abs(best.start - r.start))
    if not errs:
        return {"median": None, "p90": None, "unmatched": 1.0 if reference else 0.0}
    errs.sort()
    return {"median": round(statistics.median(errs), 2), "p90": round(errs[int(0.9 * (len(errs) - 1))], 2),
            "unmatched": round(unmatched / len(reference), 3)}


def divergence(texts_a: list[str], texts_b: list[str], threshold: float = 0.15) -> dict:
    """Compara duas gerações do mesmo conjunto de fragmentos (mesma ordem). Um fragmento é 'divergente' quando o WER entre
    as duas versões passa de `threshold` ou quando só uma delas tem texto. Base do portão de divergência do pipeline
    de qualidade: o que concorda é aceito, o que diverge vai ao revisor."""
    n = min(len(texts_a), len(texts_b))
    flagged, scores = [], []
    for i in range(n):
        a, b = texts_a[i], texts_b[i]
        if not a and not b:
            scores.append(0.0)
            continue
        score = 1.0 if (not a or not b) else min(wer(a, b), 1.0)
        scores.append(score)
        if score > threshold:
            flagged.append(i)
    return {"fragments": n, "divergent": len(flagged), "divergent_fraction": round(len(flagged) / n, 3) if n else 0.0,
            "mean_wer_between": round(sum(scores) / n, 4) if n else 0.0, "indexes": flagged}



def fragment_errors(reference: str, fragments: list[str], lang: str | None = None) -> list[dict]:
    """Atribui os erros de palavra (substituição, inserção, remoção) de uma transcrição ao FRAGMENTO onde ocorreram.

    Alinha a referência inteira com a concatenação dos fragmentos (distância de edição com retorno) e conta, por
    fragmento, os erros e as palavras de referência cobertas. Uma remoção é atribuída ao fragmento da palavra vizinha.
    Devolve [{"errors": n, "ref_words": n}, ...] na ordem dos fragmentos. Base para medir se a divergência entre dois
    modelos aponta onde há erro (precisão/cobertura do portão) e quanto um revisor perfeito ganharia."""
    r = normalize(reference, lang)
    hyp: list[str] = []
    owner: list[int] = []
    for i, t in enumerate(fragments):
        for w in normalize(t, lang):
            hyp.append(w)
            owner.append(i)
    n, m = len(r), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] != hyp[j - 1]))
    errs = [0] * len(fragments)
    refw = [0] * len(fragments)
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (r[i - 1] != hyp[j - 1]):
            o = owner[j - 1]
            refw[o] += 1
            errs[o] += r[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            errs[owner[j - 1]] += 1
            j -= 1
        else:
            o = owner[j - 1] if j > 0 else (owner[0] if owner else 0)
            if fragments:
                refw[o] += 1
                errs[o] += 1
            i -= 1
    return [{"errors": e, "ref_words": w} for e, w in zip(errs, refw)]


def gate_analysis(errors_a: list[dict], errors_b: list[dict], flagged: list[int]) -> dict:
    """Quanto vale o portão de divergência, a partir dos erros por fragmento de duas gerações (A é a prioritária).

    - precision/recall: dos fragmentos marcados como divergentes, quantos têm erro em A; dos que têm erro em A, quantos
      foram marcados;
    - oracle_all: WER se um revisor perfeito escolhesse, em TODOS os fragmentos, a melhor entre A e B;
    - oracle_gated: idem, mas só nos fragmentos marcados (o que o portão permite);
    - wer_a / wer_b: referência. Limite superior: o revisor real fica abaixo disso."""
    n = min(len(errors_a), len(errors_b))
    flagged_set = set(flagged)
    total = sum(e["ref_words"] for e in errors_a[:n]) or 1
    err_frags = {i for i in range(n) if errors_a[i]["errors"] > 0}
    tp = len(err_frags & flagged_set)
    a = sum(e["errors"] for e in errors_a[:n])
    b = sum(e["errors"] for e in errors_b[:n])
    best_all = sum(min(errors_a[i]["errors"], errors_b[i]["errors"]) for i in range(n))
    best_gated = sum(min(errors_a[i]["errors"], errors_b[i]["errors"]) if i in flagged_set else errors_a[i]["errors"] for i in range(n))
    return {"fragments": n, "fragments_with_error_a": len(err_frags), "flagged": len(flagged_set),
            "precision": round(tp / len(flagged_set), 3) if flagged_set else None,
            "recall": round(tp / len(err_frags), 3) if err_frags else None,
            "wer_a": round(a / total, 4), "wer_b": round(b / total, 4),
            "oracle_all": round(best_all / total, 4), "oracle_gated": round(best_gated / total, 4)}


def _union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(i for i in intervals if i[1] > i[0]):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def coverage(reference: list[Cue], hypothesis: list[Cue]) -> dict:
    """Cobertura temporal (independe do idioma): quanto do tempo de fala da referência a hipótese cobre (recall), quanto do tempo
    da hipótese cai onde há fala na referência (precisão) e o IoU. Mede se a legenda aparece quando a fala acontece."""
    ru = _union([(c.start, c.end) for c in reference])
    hu = _union([(c.start, c.end) for c in hypothesis])

    def overlap(xs, ys):
        i = j = 0
        total = 0.0
        while i < len(xs) and j < len(ys):
            a, b = max(xs[i][0], ys[j][0]), min(xs[i][1], ys[j][1])
            total += max(0.0, b - a)
            if xs[i][1] < ys[j][1]:
                i += 1
            else:
                j += 1
        return total

    inter = overlap(ru, hu)
    rt, ht = sum(b - a for a, b in ru), sum(b - a for a, b in hu)
    return {"recall": round(inter / rt, 4) if rt else 0.0, "precision": round(inter / ht, 4) if ht else 0.0,
            "iou": round(inter / (rt + ht - inter), 4) if rt + ht - inter else 0.0}


def chrf(reference: str, hypothesis: str, max_n: int = 6, beta: float = 2.0) -> float:
    """chrF (F-score de n-gramas de caracteres, 0–1): mede a semelhança de duas traduções; tolera variações de forma."""
    def grams(text, n):
        t = "".join(normalize(text))  # sem espaços/pontuação/acentos
        return {} if len(t) < n else __import__("collections").Counter(t[i:i + n] for i in range(len(t) - n + 1))

    precs, recs = [], []
    for n in range(1, max_n + 1):
        r, h = grams(reference, n), grams(hypothesis, n)
        if not r or not h:
            continue
        match = sum(min(c, h.get(g, 0)) for g, c in r.items())
        precs.append(match / sum(h.values()))
        recs.append(match / sum(r.values()))
    if not precs:
        return 0.0
    p, r = sum(precs) / len(precs), sum(recs) / len(recs)
    return 0.0 if p + r == 0 else (1 + beta**2) * p * r / (beta**2 * p + r)


def error_breakdown(reference: str, hypothesis: str, lang: str | None = None) -> dict:
    """WER decomposto em substituições, omissões e inserções (frações das palavras da referência) e o total de palavras da hipótese.

    A decomposição mostra O QUE está errado: inserção alta = texto inventado (ex.: música transcrita como fala); omissão alta = fala perdida."""
    r, h = normalize(reference, lang), normalize(hypothesis, lang)
    n, m = len(r), len(h)
    if n == 0:
        return {"wer": 0.0 if m == 0 else 1.0, "sub": 0.0, "del": 0.0, "ins": float(m), "hyp_words": m}
    prev = [(j, 0, 0, j) for j in range(m + 1)]  # (custo, substituições, omissões, inserções)
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            c = 0 if r[i - 1] == h[j - 1] else 1
            cur.append(min((prev[j - 1][0] + c, prev[j - 1][1] + c, prev[j - 1][2], prev[j - 1][3]),
                           (prev[j][0] + 1, prev[j][1], prev[j][2] + 1, prev[j][3]),
                           (cur[j - 1][0] + 1, cur[j - 1][1], cur[j - 1][2], cur[j - 1][3] + 1)))
        prev = cur
    cost, s, d, ins = prev[m]
    return {"wer": round(cost / n, 4), "sub": round(s / n, 4), "del": round(d / n, 4), "ins": round(ins / n, 4), "hyp_words": m}
