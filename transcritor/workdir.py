"""Pasta de trabalho dos fragmentos de áudio (opção "Qualidade"): work/<id>/ com WAVs e manifesto.json.

Regra do usuário: os fragmentos só são removidos quando o trabalho CONCLUI ou dá ERRO (nunca durante a revisão).
Cancelamento: os fragmentos ficam (decisão em aberto na proposta); `purge_old` apaga pastas antigas na inicialização.
`MT_KEEP_WORK=1` mantém a pasta mesmo ao concluir (útil para analisar a qualidade).
"""
import json
import os
import shutil
import time

from . import audio


def base_dir(config) -> str:
    return getattr(config, "work_dir", "") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work")


def create(config, job_id: str) -> str:
    path = os.path.join(base_dir(config), job_id)
    os.makedirs(path, exist_ok=True)
    return path


def save_fragments(path: str, chunks) -> None:
    import wave

    for i, ch in enumerate(chunks):
        with wave.open(os.path.join(path, f"{i:04d}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(audio.SAMPLE_RATE)
            w.writeframes(ch.samples.tobytes())


def save_manifest(path: str, frags, report) -> None:
    data = {"report": report.as_dict() if hasattr(report, "as_dict") else report,
            "fragments": [{"file": f"{f.index:04d}.wav", "start": round(f.start, 2), "end": round(f.end, 2), "a": f.a, "b": f.b, "c": f.c,
                           "flagged": f.flagged, "review": f.review, "final": f.final, "source": f.source,
                           "passes": getattr(f, "passes", []), "pre_specialists": getattr(f, "pre_specialists", "")} for f in frags]}
    with open(os.path.join(path, "manifesto.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def remove(path: str | None) -> None:
    if path and os.path.isdir(path) and os.environ.get("MT_KEEP_WORK") != "1":
        shutil.rmtree(path, ignore_errors=True)


def purge_old(config, days: float = 7.0) -> int:
    """Apaga pastas de trabalho antigas (de trabalhos cancelados/interrompidos). Devolve quantas removeu."""
    base, removed = base_dir(config), 0
    if not os.path.isdir(base):
        return 0
    limit = time.time() - days * 86400
    for name in os.listdir(base):
        p = os.path.join(base, name)
        if os.path.isdir(p) and os.path.getmtime(p) < limit:
            shutil.rmtree(p, ignore_errors=True)
            removed += 1
    return removed
