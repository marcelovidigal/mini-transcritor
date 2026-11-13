"""Valida de verdade os motores Whisper instalados com um modelo pequeno — RODE SÓ COM A GPU LIVRE.

    python tools/validate_engines.py D:/clips/fala_curta.mp3 --language pt-BR --model tiny

Para cada motor cujo pacote esteja instalado (openai-whisper, faster-whisper, whisperx), baixa o modelo escolhido (se
faltar), gera a legenda e mostra tempos, as primeiras legendas e avisos. Serve para confirmar que as chamadas às
bibliotecas reais funcionam (o app foi desenvolvido com dublês nos testes). Com --diarize testa o WhisperX com o
token do Hugging Face em HF_TOKEN.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):  # o console do Windows (cp1252) não imprime acentos/setas
    _s.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import run_one  # noqa: E402
from transcritor import engines  # noqa: E402
from transcritor.config import Config  # noqa: E402
from transcritor.ollama_client import OllamaClient  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--language", default="pt-BR")
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--diarize", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    client = OllamaClient(cfg.ollama_host, cfg.ollama_timeout)
    ok = failed = 0
    for eid, eng in engines.ENGINES.items():
        print(f"\n=== {eng.label} ({eid}) ===")
        if not eng.available():
            print(f"  pulado: {eng.status()['hint']}")
            continue
        try:
            job, text = run_one(args.audio, args.language, eid, args.model, cfg, client,
                                diarize=args.diarize and eng.supports_diarization)
            print(f"  OK em {job.wall:.1f}s · etapas: {job.timings}")
            print("  " + "\n  ".join(text.strip().split("\n")[:8]))
            for w in job.warnings:
                print("  aviso:", w)
            ok += 1
        except Exception as e:
            print(f"  FALHOU: {e.__class__.__name__}: {e}")
            failed += 1
    print(f"\nResumo: {ok} motor(es) OK, {failed} com falha.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
