"""Traduz a legenda de referência (inglês, trecho do filme) para pt-BR com o MESMO tradutor e prompt do pipeline (e4b, lotes de 8).

A tradução da referência serve de "alvo em pt-BR": ao comparar a nossa saída com ela, o efeito do tradutor se cancela e
o que sobra vem dos erros de transcrição (e das diferenças de segmentação)."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # raiz do projeto
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from transcritor import metrics, srt  # noqa: E402
from transcritor.ollama_client import OllamaClient, OllamaError  # noqa: E402

base = os.path.dirname(os.path.abspath(__file__))
cues = metrics.parse_srt(open(os.path.join(base, "ref_limpa.srt"), encoding="utf-8").read())
client = OllamaClient("http://127.0.0.1:11434", 600)
MODEL = "gemma4:12b"
out, fails, t0 = [], 0, time.time()
for i in range(0, len(cues), 8):
    batch = cues[i:i + 8]
    texts = [c.text for c in batch]
    try:
        res = client.translate_batch(MODEL, texts, "pt-BR")
    except OllamaError:
        res = None
    if res is None or len(res) != len(texts):
        fails += 1
        res = []
        for t in texts:
            try:
                res.append(client.translate(MODEL, t, "pt-BR"))
            except OllamaError:
                res.append(t)
    out += [srt.Cue(c.start, c.end, srt.clean_text(r) or c.text) for c, r in zip(batch, res)]
    if (i // 8) % 20 == 0:
        print(f"  {i}/{len(cues)} ({time.time() - t0:.0f}s)", flush=True)
open(os.path.join(base, "ref_ptbr_12b.srt"), "w", encoding="utf-8").write(srt.to_srt(out))
open(os.path.join(base, "ref_ptbr_12b.txt"), "w", encoding="utf-8").write(" ".join(c.text for c in out))
fwd = base.replace("\\", "/")
json.dump([{"audio": fwd + "/segmento_30min.mkv", "language": "pt-BR", "reference": fwd + "/ref_ptbr_12b.txt",
            "reference_src": fwd + "/ref_limpa.txt", "reference_src_srt": fwd + "/ref_limpa.srt", "language_src": "en"}],
          open(os.path.join(base, "filme_12b_descartar.json"), "w", encoding="utf-8"), indent=2)
print(f"pronto: {len(out)} legendas traduzidas em {time.time() - t0:.0f}s; lotes com falha de formato: {fails}")
