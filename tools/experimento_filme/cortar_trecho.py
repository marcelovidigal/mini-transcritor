"""Corta um trecho de um filme para o experimento: áudio exato (sem deslocamento) + legenda original recortada e limpa.

Uso: python cortar_trecho.py <arquivo-de-video> <início HH:MM:SS> <minutos> <pasta-de-saída> [idioma=en]
Saída em <pasta>: trecho.wav (16 kHz mono), ref_limpa.srt / ref_limpa.txt (a legenda ORIGINAL, sem SDH, com tempos relativos ao
trecho, sem etiquetas, sons e notas musicais) e filme.json (manifesto para tools/evaluate.py).
O áudio é recodificado (corte exato), então NÃO há o deslocamento de ~7 s do `-c copy` e não precisa de offset.json.
Nada disto vai para o repositório: a pasta de saída deve ficar fora dele."""
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # raiz do projeto
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from transcritor import metrics, srt  # noqa: E402


def secs(hms: str) -> float:
    h, m, s = (hms.split(":") + ["0", "0"])[:3] if hms.count(":") == 2 else ("0", *hms.split(":"))
    return int(h) * 3600 + int(m) * 60 + float(s)


def main():
    video, start, minutes, out = sys.argv[1], secs(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    lang = sys.argv[5] if len(sys.argv) > 5 else "en"
    os.makedirs(out, exist_ok=True)
    dur = minutes * 60
    wav = os.path.join(out, "trecho.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(dur), "-i", video, "-map", "0:a:0",
                    "-ac", "1", "-ar", "16000", wav], check=True)
    full = os.path.join(out, "legenda_completa.srt")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", video, "-map", "0:s:0", full], check=True)  # 1ª legenda = não SDH (conferir antes)
    cues = metrics.parse_srt(open(full, encoding="utf-8").read())
    keep = []
    for c in cues:
        if c.end <= start or c.start >= start + dur:
            continue
        t = c.text.replace("\n", " ")
        t = re.sub(r"<[^>]+>|\{[^}]*\}|\[[^\]]*\]|\([^)]*\)", "", t).replace("♪", "").replace("♫", "")
        t = re.sub(r"(^|\s)-\s*", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        if metrics.normalize(t):
            keep.append(srt.Cue(max(0.0, c.start - start), min(dur, c.end - start), t))
    open(os.path.join(out, "ref_limpa.srt"), "w", encoding="utf-8").write(srt.to_srt(keep))
    txt = " ".join(c.text for c in keep)
    open(os.path.join(out, "ref_limpa.txt"), "w", encoding="utf-8").write(txt)
    words = len(metrics.normalize(txt))
    cover = sum(c.end - c.start for c in keep) / dur
    print(f"cues com fala {len(keep)}; palavras {words}; {words / minutes:.0f} palavras/min; tempo coberto {cover:.0%}")
    fwd = out.replace("\\", "/")
    json.dump([{"audio": fwd + "/trecho.wav", "language": "pt-BR", "reference": fwd + "/ref_ptbr.txt",
                "reference_src": fwd + "/ref_limpa.txt", "reference_src_srt": fwd + "/ref_limpa.srt", "source_language": lang}],
              open(os.path.join(out, "filme.json"), "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
