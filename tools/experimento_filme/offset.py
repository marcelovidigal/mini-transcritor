"""Estima o deslocamento global entre o áudio do segmento e a legenda de referência.

O vídeo foi cortado com `-c copy`, que começa no keyframe ANTERIOR ao instante pedido; a legenda de referência foi cortada
exatamente no instante pedido. Resultado: os tempos das saídas são maiores que os da referência por um valor constante.
Estimativa: mediana (assinada) de início(hipótese) − início(referência) nos pares de falas que se correspondem pelas palavras,
usando só as saídas com tempo real (Whisper `turbo` e WhisperX), que são as mais precisas."""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # raiz do projeto
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from transcritor import metrics  # noqa: E402

base = os.path.dirname(os.path.abspath(__file__))
ref = metrics.parse_srt(open(os.path.join(base, "ref_limpa.srt"), encoding="utf-8").read())


def signed_offsets(src, window=15.0, min_jaccard=0.5):
    hw = [set(metrics.normalize(h.text)) for h in src]
    out = []
    for r in ref:
        rw = set(metrics.normalize(r.text))
        if len(rw) < 3:
            continue
        best, bs = None, 0.0
        for h, w in zip(src, hw):
            if not w or abs(h.start - (r.start + 7.0)) > window:
                continue
            sc = len(rw & w) / len(rw | w)
            if sc > bs:
                best, bs = h, sc
        if best is not None and bs >= min_jaccard:
            out.append(best.start - r.start)
    return out


res = {}
for name in ("turbo", "whisperx_diar"):
    src = metrics.parse_srt(open(os.path.join(base, "saidas", f"segmento_30min.{name}.src.srt"), encoding="utf-8").read())
    off = signed_offsets(src)
    res[name] = {"n": len(off), "median": round(statistics.median(off), 3), "mean": round(statistics.mean(off), 3),
                 "p25": round(sorted(off)[len(off) // 4], 3), "p75": round(sorted(off)[3 * len(off) // 4], 3)}
    print(name, res[name])
final = round(statistics.mean(r["median"] for r in res.values()), 2)
print("deslocamento adotado (s):", final)
json.dump({"offset_s": final, "detail": res}, open(os.path.join(base, "offset.json"), "w"), indent=1)
