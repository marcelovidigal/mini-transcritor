"""Juiz de fidelidade: compara cada fala da legenda ORIGINAL (inglês) com o que a nossa legenda (pt-BR) mostra naquele instante.

Nota por fala: 2 = o português diz o mesmo (variações de redação são aceitas); 1 = parcial (falta ou muda parte do sentido);
0 = errado, sem relação ou ausente. O juiz (gemma4:12b, só texto) não sabe qual sistema gerou a saída (avaliação cega).
Uso: python judge.py  (lê saidas/segmento_30min.<sistema>.srt; escreve judge_resultado.json)"""
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # raiz do projeto
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from transcritor import metrics  # noqa: E402
from transcritor.ollama_client import OllamaClient, OllamaError  # noqa: E402

base = os.path.dirname(os.path.abspath(__file__))
SYSTEMS = sys.argv[1:] or ["e4b", "turbo", "whisperx_diar", "qualidade"]
N, BATCH, MODEL = 160, 5, "gemma4:12b"
ref = [c for c in metrics.parse_srt(open(os.path.join(base, "ref_trim.srt"), encoding="utf-8").read()) if len(metrics.normalize(c.text)) >= 4]
random.seed(11)
sample = sorted(random.sample(ref, min(N, len(ref))), key=lambda c: c.start)
client = OllamaClient("http://127.0.0.1:11434", 900)
SCHEMA = {"type": "object", "properties": {"scores": {"type": "array", "items": {"type": "integer", "enum": [0, 1, 2]}}}, "required": ["scores"]}


WIN = float(os.environ.get('JUDGE_WINDOW', '0.4'))  # tolerância temporal ao alinhar fala original e legenda gerada (s)
SUFFIX = os.environ.get('JUDGE_SUFFIX', '')
OFFSET = json.load(open(os.path.join(base, "offset.json")))["offset_s"]  # a saída está adiantada em relação à referência (corte no keyframe)


def hyp_text(cues, c):
    parts = [h.text.replace("\n", " ") for h in cues if h.end - OFFSET > c.start - WIN and h.start - OFFSET < c.end + WIN]
    return " ".join(parts)


def judge(pairs):
    lines = "\n".join(f'{i + 1}. ORIGINAL (English): {o}\n   SUBTITLE (Portuguese): {p or "(nothing shown)"}' for i, (o, p) in enumerate(pairs))
    prompt = ("You evaluate a Portuguese (pt-BR) subtitle against the original English line spoken at the same moment. The Portuguese "
              "text may also include words from neighbouring lines: ignore that. For each numbered item give a score: 2 = the Portuguese "
              "conveys the same meaning as the original (wording differences are fine); 1 = partially (part of the meaning is missing or "
              "changed); 0 = wrong, unrelated, or nothing shown. Return JSON {\"scores\": [one integer per item, in order]}.\n\n" + lines)
    raw = client._chat({"model": MODEL, "stream": False, "think": False, "options": {"temperature": 0}, "format": SCHEMA,
                        "messages": [{"role": "user", "content": prompt}]})
    scores = json.loads(raw)["scores"]
    if len(scores) != len(pairs):
        raise ValueError("tamanho")
    return scores


results = {}
for name in SYSTEMS:
    path = os.path.join(base, "saidas", f"segmento_30min.{name}.srt")
    if not os.path.exists(path):
        print("sem saída:", name)
        continue
    cues = metrics.parse_srt(open(path, encoding="utf-8").read())
    pairs = [(c.text, hyp_text(cues, c)) for c in sample]
    scores, t0 = [], time.time()
    for i in range(0, len(pairs), BATCH):
        chunk = pairs[i:i + BATCH]
        try:
            scores += judge(chunk)
        except (OllamaError, ValueError, KeyError, json.JSONDecodeError):
            for pr in chunk:  # fallback: um a um
                try:
                    scores += judge([pr])
                except Exception:
                    scores.append(0)
    n = len(scores)
    results[name] = {"n": n, "mean": round(sum(scores) / n / 2, 4), "faithful_2": round(scores.count(2) / n, 4),
                     "partial_1": round(scores.count(1) / n, 4), "wrong_0": round(scores.count(0) / n, 4),
                     "empty": sum(1 for _o, p in pairs if not p), "scores": scores, "seconds": round(time.time() - t0)}
    print(f"{name:14} nota média {results[name]['mean']:.3f} (0–1) | fiel {results[name]['faithful_2']:.0%} parcial {results[name]['partial_1']:.0%} "
          f"errada/ausente {results[name]['wrong_0']:.0%} | sem texto {results[name]['empty']} | {results[name]['seconds']}s", flush=True)
json.dump({"model": MODEL, "sample": len(sample), "systems": results}, open(os.path.join(base, "judge_resultado" + SUFFIX + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
