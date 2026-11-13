"""Votação entre hipóteses do MESMO modelo de áudio (modo de qualidade 1, sem WhisperX): consenso e pontos de dúvida.

O autorrefinamento sozinho não corrige substituições (o revisor da mesma família ouve os mesmos erros que o gerador). Aqui o gerador
transcreve cada fragmento 3 vezes com instruções e áudio diferentes (padrão, literal, atento com ganho normalizado); o código alinha as
três palavra a palavra e:

  * **consenso:** onde 2 de 3 concordam numa troca, ela vale; o texto-base é a 1ª hipótese (a padrão);
  * **dúvidas:** onde só uma das outras discorda, o ponto vira uma pergunta objetiva para o revisor (atual × outra); quem decide é o
    revisor com o áudio, e o texto final é MONTADO pelo código (nada de texto livre);
  * **estabilidade:** concordância média entre as hipóteses; muito baixa costuma ser música/efeito, e o revisor é avisado.
"""
import difflib
import re

from . import judge_v2

MAX_DOUBTS = 10
UNSTABLE_BELOW = 0.5  # concordância média abaixo disto = hipóteses muito diferentes
_KEY = re.compile(r"\w+", re.UNICODE)


def _k(tok: str) -> str:
    return "".join(_KEY.findall(tok)).casefold()


def agreement(a: str, b: str) -> float:
    """Concordância (0–1) entre dois textos: razão de difflib sobre as palavras normalizadas; dois textos vazios concordam."""
    ak, bk = [_k(t) for t in a.split()], [_k(t) for t in b.split()]
    if not ak and not bk:
        return 1.0
    if not ak or not bk:
        return 0.0
    return difflib.SequenceMatcher(None, ak, bk, autojunk=False).ratio()


def _proposals(pivot: list[str], other: str) -> dict[tuple[int, int], list[str]]:
    """Trocas que `other` propõe sobre o pivô: {(i1, i2): palavras que entram no lugar de pivot[i1:i2]}."""
    ot = other.split()
    sm = difflib.SequenceMatcher(None, [_k(t) for t in pivot], [_k(t) for t in ot], autojunk=False)
    return {(i1, i2): ot[j1:j2] for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"}


def consensus(hyps: list[str]) -> dict:
    """{"text": consenso, "alt": texto com a primeira alternativa de cada dúvida, "stability": 0–1, "applied": trocas por maioria,
    "doubts": nº de pontos em dúvida}. `hyps[0]` é a hipótese padrão (pivô quando não vazia)."""
    hyps = [h.strip() for h in hyps]
    filled = [h for h in hyps if h]
    if len(filled) * 2 < len(hyps):  # a maioria ouviu silêncio
        return {"text": "", "alt": "", "stability": 1.0 if not filled else 0.0, "applied": 0, "doubts": 0}
    pivot_text = filled[0]
    others = filled[1:]
    pivot = pivot_text.split()
    props = [_proposals(pivot, o) for o in others]
    pair = [agreement(pivot_text, o) for o in others] + ([agreement(others[0], others[1])] if len(others) == 2 else [])
    stability = sum(pair) / len(pair) if pair else 1.0
    regions = sorted({r for p in props for r in p})
    new_toks: dict[tuple[int, int], list[str]] = {}  # trocas por maioria
    alt_toks: dict[tuple[int, int], list[str]] = {}  # primeira alternativa das dúvidas
    for r in regions:
        cands = [p[r] for p in props if r in p]
        keys = [tuple(_k(t) for t in c) for c in cands]
        if len(cands) >= 2 and len(set(keys)) == 1:
            new_toks[r] = cands[0]
        else:  # só uma das outras discorda (ou discordam entre si): dúvida
            alt_toks[r] = cands[0]

    def build(base: dict, extra: dict) -> str:
        out, pos = [], 0
        for (i1, i2) in sorted({**base, **extra}):
            if i1 < pos:  # região sobreposta a outra já aplicada: ignora
                continue
            out += pivot[pos:i1] + {**base, **extra}[(i1, i2)]
            pos = i2
        return " ".join(out + pivot[pos:])

    text = build(new_toks, {})
    alt = build(new_toks, alt_toks)
    return {"text": text, "alt": alt, "stability": round(stability, 3), "applied": len(new_toks),
            "doubts": len(judge_v2.word_diffs(alt, text)) if alt != text else 0}


def doubts_of(text: str, alt: str) -> list[dict]:
    """Pontos para o revisor: [{"current": trecho do consenso, "other": trecho alternativo}] (um lado pode ser vazio)."""
    return [{"current": d["b"], "other": d["a"]} for d in judge_v2.word_diffs(alt, text)]


def apply_choices(text: str, alt: str, choices: list) -> str:
    """Texto final: base = consenso; em cada ponto vale a alternativa quando `choices[k]` for "other"."""
    picks = ["A" if str(c).strip().lower() == "other" else "B" for c in choices]
    return judge_v2.compose(alt, text, picks)
