import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # raiz do projeto
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from transcritor import metrics  # noqa: E402

base = os.path.dirname(os.path.abspath(__file__))
OFF = json.load(open(os.path.join(base, "offset.json")))["offset_s"]
ref = metrics.parse_srt(open(os.path.join(base, "ref_limpa.srt"), encoding="utf-8").read())
print("palavras na referência:", sum(len(metrics.normalize(c.text)) for c in ref))
for name in ("e4b", "turbo", "whisperx_diar", "qualidade"):
    src = metrics.parse_srt(open(os.path.join(base, "saidas", f"segmento_30min.{name}.src.srt"), encoding="utf-8").read())
    hw = sum(len(metrics.normalize(c.text)) for c in src)
    # recall por janela de 3 min: palavras da referência cujo texto aparece nas falas da saída dentro de ±8 s
    row = []
    for w0 in range(0, 1800, 180):
        rr = [c for c in ref if w0 <= c.start < w0 + 180]
        hh = [h for h in src if w0 - 20 <= h.start - OFF < w0 + 200]
        hset = set(w for h in hh for w in metrics.normalize(h.text))
        rwords = [w for c in rr for w in metrics.normalize(c.text)]
        row.append(round(sum(1 for w in rwords if w in hset) / max(len(rwords), 1) * 100))
    print(f"{name:14} palavras na saída {hw:5d} | cobertura de vocabulário da referência por janela de 3 min (%): {row}")
