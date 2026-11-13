"""Avaliação de velocidade e qualidade — RODE SÓ COM A GPU LIVRE (usa o Ollama / os motores de verdade).

Exemplo:
    python tools/evaluate.py manifest.json --engine ollama --model gemma4:e4b --parallel 1 2 4 --out resultado.json

O manifesto é uma lista JSON; cada item:
    {"audio": "D:/clips/aula.mp3", "language": "pt-BR", "reference": "D:/clips/aula.txt", "reference_srt": "D:/clips/aula.srt"}
`reference` (texto corrido, no idioma de SAÍDA) habilita o WER; `reference_srt` habilita o erro de tempo. Ambos opcionais.

Para cada combinação (clipe × valor de --parallel × --translate-batch) mede: tempo por etapa, RTF (tempo de
processamento / duração do áudio), WER, violações das regras de legenda e erro de tempo. Gera tabela e JSON.
"""
import argparse
import itertools
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):  # o console do Windows (cp1252) não imprime acentos/setas
    _s.reconfigure(encoding="utf-8", errors="replace")

from transcritor import metrics, pipeline  # noqa: E402
from transcritor.config import Config  # noqa: E402
from transcritor.jobs import Job  # noqa: E402
from transcritor.ollama_client import OllamaClient  # noqa: E402


def run_one(audio_path, language, engine, model, config, client, diarize=False, context=""):
    """Executa o pipeline de verdade num arquivo e devolve (job, texto do SRT)."""
    out_dir = tempfile.mkdtemp(prefix="mt_eval_")
    job = Job(id="eval-" + os.path.splitext(os.path.basename(audio_path))[0], path=audio_path, language=language, model=model, engine=engine, diarize=diarize,
              context=context, source_name=os.path.basename(audio_path), output_dir=out_dir)
    t0 = time.perf_counter()
    pipeline.run_job(job, client, config)
    job.wall = time.perf_counter() - t0
    with open(job.output_path, encoding="utf-8") as f:
        return job, f.read()


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def evaluate(item, engine, model, parallel, batch, args):
    cfg = Config(translate_model=args.translate_model, parallel=parallel, translate_batch=batch, chunk_seconds=args.chunk, vad=not args.no_vad,
                 quality_b_engine=args.quality_b_engine, quality_b_model=args.quality_b_model,
                 quality_reviewer=args.quality_reviewer, quality_threshold=args.quality_threshold,
                 quality_prefer=args.quality_prefer, quality_c_engine=args.quality_c_engine,
                 quality_c_model=args.quality_c_model, quality_loops=args.quality_loops,
                 quality_specialists=args.quality_specialists)
    client = OllamaClient(cfg.ollama_host, cfg.ollama_timeout)
    job, text = run_one(item["audio"], item["language"], engine, model, cfg, client, args.diarize, args.context)
    cues = metrics.parse_srt(text)
    if args.save_srt:  # guarda a legenda gerada para comparações posteriores
        os.makedirs(args.save_srt, exist_ok=True)
        stem = os.path.splitext(os.path.basename(item["audio"]))[0]
        with open(os.path.join(args.save_srt, f"{stem}.{args.label or engine}.srt"), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    row = {"audio": os.path.basename(item["audio"]), "engine": engine, "model": model, "parallel": parallel,
           "translate_batch": batch, "wall_s": round(job.wall, 1), "audio_s": round(job.audio_seconds, 1),
           "rtf": round(job.wall / job.audio_seconds, 3) if job.audio_seconds else None,
           "stages_s": job.timings, "rules": metrics.rule_stats(cues), "warnings": job.warnings, "report": job.report,
           "speakers": sorted({m for c in cues for m in __import__("re").findall(r"\[Falante (\d+)\]", c.text)}),
           "turns": sum(1 for c in cues if "[Falante" in c.text)}
    if args.save_srt and job.source_cues:  # texto na língua falada, antes da tradução
        from transcritor import srt as _srt

        src = [_srt.Cue(s, e, t) for s, e, t in job.source_cues]
        stem = os.path.splitext(os.path.basename(item["audio"]))[0]
        with open(os.path.join(args.save_srt, f"{stem}.{args.label or engine}.src.srt"), "w", encoding="utf-8", newline="\n") as f:
            f.write(_srt.to_srt(src))
        if item.get("reference_src"):
            row["wer_src"] = round(metrics.wer(read(item["reference_src"]), metrics.plain_text(src), item.get("language_src", "en")), 4)
        if item.get("reference_src_srt"):
            row["timing_src"] = metrics.timing_error(metrics.parse_srt(read(item["reference_src_srt"])), src)
    if item.get("reference_src_srt"):
        row["coverage"] = metrics.coverage(metrics.parse_srt(read(item["reference_src_srt"])), cues)
    if item.get("reference"):
        row["wer"] = round(metrics.wer(read(item["reference"]), metrics.plain_text(cues), item["language"]), 4)
    if item.get("reference_srt"):
        row["timing"] = metrics.timing_error(metrics.parse_srt(read(item["reference_srt"])), cues)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--engine", default="ollama")
    ap.add_argument("--model", required=True)
    ap.add_argument("--parallel", type=int, nargs="+", default=[1])
    ap.add_argument("--translate-batch", type=int, nargs="+", default=[8])
    ap.add_argument("--chunk", type=float, default=12.0)
    ap.add_argument("--no-vad", action="store_true")
    ap.add_argument("--diarize", action="store_true")
    ap.add_argument("--context", default="")
    ap.add_argument("--translate-model", default="gemma4:e4b", help="modelo do Ollama que traduz quando o motor não é o Ollama")
    ap.add_argument("--save-srt", default="", help="pasta onde guardar a legenda gerada")
    ap.add_argument("--label", default="", help="nome do sistema no arquivo salvo (padrão: o motor)")
    ap.add_argument("--quality-b-engine", default="openai-whisper", help="motor Qualidade: 2ª geração (openai-whisper, faster-whisper… ou ollama)")
    ap.add_argument("--quality-b-model", default="large-v3-turbo")
    ap.add_argument("--quality-reviewer", default="gemma4:12b", help="modelo do Ollama que revisa (padrão do app: gemma4:12b; vazio = o mesmo da 1ª geração)")
    ap.add_argument("--quality-threshold", type=float, default=0.05)
    ap.add_argument("--quality-prefer", choices=["A", "B"], default="B", help="geração prioritária quando o revisor não decide")
    ap.add_argument("--quality-c-engine", default="", help="3ª geração opcional: openai-whisper, faster-whisper… ou ollama")
    ap.add_argument("--quality-c-model", default="")
    ap.add_argument("--quality-loops", type=int, default=0, help="passadas de refinamento depois da 1ª (os modos quality-e4b/-12b já usam 2)")
    ap.add_argument("--quality-specialists", action="store_true", help="especialistas: termos, pontuação e revisão da tradução")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    items = json.loads(read(args.manifest))
    rows = []
    for item, par, bat in itertools.product(items, args.parallel, args.translate_batch):
        print(f"→ {os.path.basename(item['audio'])}  parallel={par} batch={bat} ...", flush=True)
        try:
            rows.append(evaluate(item, args.engine, args.model, par, bat, args))
        except Exception as e:  # uma falha não derruba a rodada inteira
            rows.append({"audio": os.path.basename(item["audio"]), "parallel": par, "translate_batch": bat, "error": str(e)})
    print(f"\n{'clipe':24} {'par':>3} {'lote':>4} {'seg':>7} {'RTF':>6} {'WER':>6} {'cps>':>5} {'linha>':>6} {'t.med':>6}")
    for r in rows:
        if "error" in r:
            print(f"{r['audio']:24} {r['parallel']:>3} {r['translate_batch']:>4}  ERRO: {r['error']}")
            continue
        t = (r.get("timing") or {}).get("median")
        print(f"{r['audio']:24} {r['parallel']:>3} {r['translate_batch']:>4} {r['wall_s']:>7} {r['rtf']!s:>6} "
              f"{r.get('wer', '-')!s:>6} {r['rules']['over_cps']:>5} {r['rules']['line_too_long']:>6} {t if t is not None else '-'!s:>6}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"\nResultado completo em {args.out}")


if __name__ == "__main__":
    main()
