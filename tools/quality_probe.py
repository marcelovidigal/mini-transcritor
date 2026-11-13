"""Sonda da ideia "Qualidade" (duas gerações + portão de divergência) — RODE SÓ COM A GPU LIVRE.

    python tools/quality_probe.py D:/clips/aula.mp3 --language pt-BR --model-a gemma4:e4b --model-b gemma4:e2b ^
        --reference D:/clips/aula.txt --keep-fragments D:/clips/frag --out sonda.json

Pergunta que responde (com poucos minutos de áudio): se dois modelos de áudio do Ollama transcrevem os mesmos
fragmentos, QUANTOS divergem e o quanto a divergência denuncia erro? Para isso:
  1. extrai o áudio e corta em fragmentos de fala (mesmo VAD do app);
  2. transcreve todos os fragmentos com o modelo A e DEPOIS com o B (em fases: GPU única, sem trocar de modelo a cada trecho);
  3. limpa o texto (quality.py) e mede a divergência entre A e B por fragmento (metrics.divergence);
  4. com --reference, mede o WER de A e de B (no idioma falado; não traduz), atribui os erros a cada fragmento e calcula
     a PRECISÃO e a COBERTURA do portão (os fragmentos divergentes são os que têm erro?) e o ganho máximo de um revisor
     perfeito (oracle_all: escolhendo a melhor versão em todos os fragmentos; oracle_gated: só nos divergentes).
Os fragmentos podem ser guardados em WAV (--keep-fragments) para o revisor do pipeline reouvir depois.
"""
import argparse
import json
import os
import sys
import tempfile
import time
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):  # o console do Windows (cp1252) não imprime acentos/setas
    _s.reconfigure(encoding="utf-8", errors="replace")

from transcritor import audio, metrics, quality, srt  # noqa: E402
from transcritor.config import Config  # noqa: E402
from transcritor.ollama_client import OllamaClient  # noqa: E402


def transcribe_all(client, model, chunks):
    """Uma fase: todos os fragmentos num único modelo. Devolve (textos, segundos)."""
    t0 = time.perf_counter()
    texts = []
    for i, ch in enumerate(chunks, 1):
        try:
            texts.append(srt.clean_text(client.transcribe(model, audio.to_wav_bytes(ch.samples))))
        except Exception as e:  # um fragmento ruim não derruba a sonda
            print(f"  fragmento {i}: {e}")
            texts.append("")
        if i % 10 == 0:
            print(f"  {model}: {i}/{len(chunks)}", flush=True)
    return texts, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--model-a", required=True)
    ap.add_argument("--model-b", required=True)
    ap.add_argument("--language", default="pt-BR", help="só informativo: a sonda não traduz")
    ap.add_argument("--reference", default="", help="texto correto (idioma falado), para calcular o WER")
    ap.add_argument("--threshold", type=float, default=0.15, help="WER entre A e B acima do qual o fragmento é divergente")
    ap.add_argument("--chunk", type=float, default=12.0)
    ap.add_argument("--keep-fragments", default="", help="pasta onde guardar os fragmentos em WAV + manifesto")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = Config(chunk_seconds=args.chunk)
    client = OllamaClient(cfg.ollama_host, cfg.ollama_timeout)
    with tempfile.TemporaryDirectory(prefix="mt_probe_") as tmp:
        wav = os.path.join(tmp, "a.wav")
        audio.extract_wav(args.audio, wav)
        samples = audio.read_wav(wav)
    chunks = audio.split_by_speech(samples, cfg.chunk_seconds, cfg.max_chunk_seconds)
    print(f"{len(chunks)} fragmentos de fala em {len(samples) / audio.SAMPLE_RATE:.0f} s de áudio")

    ta, sa = transcribe_all(client, args.model_a, chunks)
    tb, sb = transcribe_all(client, args.model_b, chunks)
    durations = [ch.end - ch.start for ch in chunks]
    ta, _ = quality.clean_texts(ta, durations)
    tb, _ = quality.clean_texts(tb, durations)

    div = metrics.divergence(ta, tb, args.threshold)
    result = {"audio": os.path.basename(args.audio), "model_a": args.model_a, "model_b": args.model_b, "threshold": args.threshold,
              "seconds_a": round(sa, 1), "seconds_b": round(sb, 1), "divergence": div}
    if args.reference:
        with open(args.reference, encoding="utf-8") as f:
            ref = f.read()
        result["wer_a"] = round(metrics.wer(ref, " ".join(ta), args.language), 4)
        result["wer_b"] = round(metrics.wer(ref, " ".join(tb), args.language), 4)
        # erros atribuídos a cada fragmento (alinhamento com a referência inteira) -> o portão aponta onde há erro?
        fe_a, fe_b = metrics.fragment_errors(ref, ta, args.language), metrics.fragment_errors(ref, tb, args.language)
        result["gate"] = metrics.gate_analysis(fe_a, fe_b, div["indexes"])

    print(json.dumps({k: v for k, v in result.items() if k != "divergence"}, ensure_ascii=False, indent=2))
    print(f"divergentes: {div['divergent']}/{div['fragments']} ({div['divergent_fraction']:.0%}), WER médio entre A e B: {div['mean_wer_between']}")

    if args.keep_fragments:
        os.makedirs(args.keep_fragments, exist_ok=True)
        manifest = []
        for i, ch in enumerate(chunks):
            name = f"{i:04d}.wav"
            with wave.open(os.path.join(args.keep_fragments, name), "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(audio.SAMPLE_RATE)
                w.writeframes(ch.samples.tobytes())
            manifest.append({"file": name, "start": round(ch.start, 2), "end": round(ch.end, 2), "a": ta[i], "b": tb[i],
                             "divergent": i in div["indexes"]})
        with open(os.path.join(args.keep_fragments, "manifesto.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"Fragmentos e manifesto em {args.keep_fragments}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
