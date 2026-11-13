"""Juiz cego de fidelidade: compara cada fala da legenda ORIGINAL (inglês) com o que a nossa legenda (pt-BR) mostra naquele instante.

    python juiz_fidelidade.py <pasta com ref_limpa.srt> <saida1.srt> [<saida2.srt> ...] [--janela 0.4] [--amostra 160] [--out juiz.json]

Nota por fala: 2 = o português diz o mesmo (variações de redação são aceitas); 1 = parcial; 0 = errado, sem relação ou ausente.
O juiz (gemma4:12b, só texto) NÃO sabe qual sistema gerou a saída (cada sistema é avaliado separadamente, mesma amostra de falas, mesma semente).
O áudio do trecho foi cortado de forma exata, então não há deslocamento entre a legenda original e a saída (sem offset.json).
Usa a GPU: rode com o Ollama livre. Só métricas saem daqui (nenhuma fala é impressa)."""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # raiz do projeto
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from transcritor import metrics  # noqa: E402
from transcritor.ollama_client import OllamaClient, OllamaError  # noqa: E402

MODEL, BATCH = "gemma4:12b", 5
SCHEMA = {"type": "object", "properties": {"scores": {"type": "array", "items": {"type": "integer", "enum": [0, 1, 2]}}}, "required": ["scores"]}


def hyp_text(cues, c, window):
    return " ".join(h.text.replace("\n", " ") for h in cues if h.end > c.start - window and h.start < c.end + window)


def judge(client, pairs):
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pasta")
    ap.add_argument("saidas", nargs="+")
    ap.add_argument("--janela", type=float, default=0.4, help="tolerância temporal (s): 0,4 = estrita; 4 = tolerante")
    ap.add_argument("--amostra", type=int, default=160)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    ref = [c for c in metrics.parse_srt(open(os.path.join(a.pasta, "ref_limpa.srt"), encoding="utf-8").read()) if len(metrics.normalize(c.text)) >= 4]
    random.seed(11)
    sample = sorted(random.sample(ref, min(a.amostra, len(ref))), key=lambda c: c.start)
    client = OllamaClient("http://127.0.0.1:11434", 900)
    results = {}
    for path in a.saidas:
        name = os.path.basename(path)
        cues = metrics.parse_srt(open(path, encoding="utf-8").read())
        pairs = [(c.text, hyp_text(cues, c, a.janela)) for c in sample]
        scores, t0 = [], time.time()
        for i in range(0, len(pairs), BATCH):
            chunk = pairs[i:i + BATCH]
            try:
                scores += judge(client, chunk)
            except (OllamaError, ValueError, KeyError, json.JSONDecodeError):
                for pr in chunk:  # fallback: um a um
                    try:
                        scores += judge(client, [pr])
                    except Exception:
                        scores.append(0)
        n = len(scores)
        results[name] = {"n": n, "mean": round(sum(scores) / n / 2, 4), "faithful_2": round(scores.count(2) / n, 4),
                         "partial_1": round(scores.count(1) / n, 4), "wrong_0": round(scores.count(0) / n, 4),
                         "empty": sum(1 for _o, p in pairs if not p), "scores": scores, "seconds": round(time.time() - t0)}
        r = results[name]
        print(f"{name:60} nota {r['mean']:.3f} | fiel {r['faithful_2']:.0%} parcial {r['partial_1']:.0%} errada {r['wrong_0']:.0%} | sem texto {r['empty']} | {r['seconds']}s", flush=True)
    if a.out:
        json.dump({"model": MODEL, "janela": a.janela, "amostra": len(sample), "systems": results}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
