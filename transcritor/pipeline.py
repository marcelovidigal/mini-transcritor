"""Pipeline: (baixar modelo) → validar → extrair áudio → dividir → transcrever → gravar SRT."""
import os
import tempfile

from . import audio, media, srt
from .catalog import norm
from .config import Config
from .jobs import Job
from .media import MediaError
from .ollama_client import OllamaClient, OllamaError

PULL_SHARE = 30.0  # % da barra reservados ao download, quando o modelo escolhido ainda não está instalado


def output_path_for(src: str, language: str) -> str:
    stem = os.path.splitext(src)[0]
    return f"{stem}.{language}.srt"


def _safe_stem(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name or "legenda"))[0]
    stem = "".join(c for c in stem if c.isprintable() and c not in '<>:"/\\|?*').strip(" .")
    return stem[:120] or "legenda"


def _write_srt(job: Job, content: str, config: Config) -> str:
    if job.output_dir:  # modo servidor: pasta download/, nome interno = id do job (sem colisões)
        os.makedirs(job.output_dir, exist_ok=True)
        job.download_name = f"{_safe_stem(job.source_name)}.{job.language}.srt"
        target = os.path.join(job.output_dir, f"{job.id}.srt")
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        return target
    target = output_path_for(job.path, job.language)
    try:
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        return target
    except OSError:
        os.makedirs(config.output_fallback_dir, exist_ok=True)
        fallback = os.path.join(config.output_fallback_dir, os.path.basename(target))
        with open(fallback, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        job.warnings.append(f"Sem permissão de escrita na pasta de origem; legenda salva em {fallback}")
        return fallback


def _ensure_model(job: Job, client: OllamaClient) -> float:
    """Baixa o modelo se não estiver instalado. Devolve o deslocamento (%) da barra para as etapas seguintes."""
    if norm(job.model) in {norm(m["name"]) for m in client.audio_models()}:
        return 0.0
    job.update(0, f"Baixando o modelo {job.model}")

    def on_progress(done, total, status):
        job.check_cancel()
        if total:
            job.update(PULL_SHARE * done / total, f"Baixando o modelo {job.model}",
                       f"{done / 1e9:.2f} de {total / 1e9:.2f} GB")

    client.pull(job.model, on_progress)
    job.check_cancel()
    if norm(job.model) not in {norm(m["name"]) for m in client.audio_models()}:
        raise OllamaError(f"O modelo '{job.model}' foi baixado, mas não declara suporte a áudio.")
    job.message = ""
    return PULL_SHARE


def run_job(job: Job, client: OllamaClient, config: Config) -> None:
    lo = _ensure_model(job, client)

    def upd(p, stage=None, message=None):  # mapeia 0–100 para [lo, 100]
        job.update(lo + p * (100 - lo) / 100, stage, message)

    upd(0, "Validando arquivo")
    info = media.probe(job.path)
    job.check_cancel()
    limit = getattr(config, "max_duration_min", 0)
    if limit and info.duration > limit * 60:
        raise MediaError(f"A mídia tem {info.duration / 60:.0f} min; o limite deste servidor é {limit:.0f} min.")
    upd(5, "Extraindo áudio do vídeo" if info.kind == "video" else "Preparando áudio")

    with tempfile.TemporaryDirectory(prefix="minitranscritor_") as tmp:
        wav = os.path.join(tmp, "audio.wav")
        audio.extract_wav(job.path, wav)
        job.check_cancel()
        samples = audio.read_wav(wav)

    upd(15, "Dividindo em trechos")
    chunks = audio.split_chunks(samples, config.chunk_seconds, config.max_chunk_seconds)
    if not chunks:
        raise MediaError("Nenhuma fala detectada no áudio (arquivo silencioso).")

    cues: list[srt.Cue] = []
    failures = 0
    last_error = ""
    total = len(chunks)
    for i, ch in enumerate(chunks, 1):
        job.check_cancel()
        upd(15 + 82 * (i - 1) / total, f"Transcrevendo trecho {i} de {total}")
        try:
            text = srt.clean_text(client.transcribe(job.model, audio.to_wav_bytes(ch.samples)))
            if text:
                try:
                    translated = srt.clean_text(client.translate(job.model, text, job.language))
                    text = translated or text
                except OllamaError as e:  # mantém o texto original; segue o fluxo
                    job.warnings.append(f"Trecho {i}: tradução falhou, mantido o idioma original ({e})")
        except OllamaError as e:
            failures += 1
            last_error = str(e)
            job.warnings.append(f"Trecho {i} ({srt.format_timestamp(ch.start)}) falhou: {e}")
            if failures >= 3 and failures == i:  # as primeiras tentativas falham: servidor/modelo inviável
                raise OllamaError(last_error)
            continue
        cues.extend(srt.cues_for_chunk(ch.start, ch.end, text))
        upd(15 + 82 * i / total)

    if failures == total:
        raise OllamaError(last_error or "Todos os trechos falharam.")
    if not cues:
        raise MediaError("O modelo não retornou texto para nenhum trecho.")

    upd(97, "Gravando legenda")
    job.output_path = _write_srt(job, srt.to_srt(cues), config)
    job.message = f"{len(cues)} legendas geradas"
