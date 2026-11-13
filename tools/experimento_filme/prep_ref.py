import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # raiz do projeto
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from transcritor import metrics, srt  # noqa: E402

base = os.path.dirname(os.path.abspath(__file__))
fwd = base.replace("\\", "/")
cues = metrics.parse_srt(open(os.path.join(base, "referencia.srt"), encoding="utf-8").read())
out = []
for c in cues:
    t = c.text.replace("\n", " ")
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"\{[^}]*\}", "", t)
    t = re.sub(r"\[[^\]]*\]", "", t)
    t = re.sub(r"\([^)]*\)", "", t)
    t = t.replace("♪", "").replace("♫", "")
    t = re.sub(r"(^|\s)-\s*", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if metrics.normalize(t):
        out.append(srt.Cue(c.start, c.end, t))
open(os.path.join(base, "ref_limpa.srt"), "w", encoding="utf-8").write(srt.to_srt(out))
txt = " ".join(c.text for c in out)
open(os.path.join(base, "ref_limpa.txt"), "w", encoding="utf-8").write(txt)
dur = 1801.4
words = len(metrics.normalize(txt))
print(f"cues originais {len(cues)} -> com fala {len(out)}; palavras {words}; {words / dur * 60:.0f} palavras/min; "
      f"tempo coberto por legendas {sum(c.end - c.start for c in out) / dur:.0%}")
json.dump([{"audio": fwd + "/segmento_30min.mkv", "language": "en", "reference": fwd + "/ref_limpa.txt",
            "reference_srt": fwd + "/ref_limpa.srt"}], open(os.path.join(base, "filme.json"), "w", encoding="utf-8"), indent=2)
