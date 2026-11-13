"""Pipeline: (baixar modelo) → validar → extrair áudio → dividir → transcrever → gravar SRT."""
import logging
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import audio, engines, langid, media, quality, quality_engine, srt, workdir
from .catalog import norm
from .engines import Engine, EngineError
from .config import Config
from .jobs import Cancelled, Job
from .media import MediaError
from .ollama_client import OllamaClient, OllamaError

joblog = logging.getLogger("transcritor.job")

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


def _ensure_model(job: Job, client: OllamaClient, model: str | None = None) -> float:
    """Baixa o modelo se não estiver instalado. Devolve o deslocamento (%) da barra para as etapas seguintes."""
    model = model or job.model
    if norm(model) in {norm(m["name"]) for m in client.audio_models()}:
        return 0.0
    job.update(0, f"Baixando o modelo {model}")

    def on_progress(done, total, status):
        job.check_cancel()
        if total:
            job.update(PULL_SHARE * done / total, f"Baixando o modelo {model}",
                       f"{done / 1e9:.2f} de {total / 1e9:.2f} GB")

    client.pull(model, on_progress)
    job.check_cancel()
    if norm(job.model) not in {norm(m["name"]) for m in client.audio_models()}:
        raise OllamaError(f"O modelo '{job.model}' foi baixado, mas não declara suporte a áudio.")
    job.message = ""
    return PULL_SHARE


TRANSCRIBE_END = 75.0  # % da barra (já remapeada) ao fim da transcrição; o resto é tradução
MAX_PARALLEL = 4


class _Stopwatch:
    """Cronometra as etapas do trabalho (segundos) em `job.timings`."""

    def __init__(self, store: dict):
        self.store, self.t = store, time.perf_counter()

    def lap(self, name: str):
        now = time.perf_counter()
        self.store[name] = round(self.store.get(name, 0.0) + now - self.t, 2)
        self.t = now


def format_hms(seconds: float) -> str:
    """Duração no formato hh:mm:ss (ex.: 4379 s → '01:12:59')."""
    s = int(round(max(seconds, 0)))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def timing_summary(timings: dict, audio_seconds: float) -> str:
    """'42 s para 600 s de áudio (RTF 0.07: 14x mais rápido que o tempo real)'. RTF = tempo de processamento / duração."""
    total = sum(v for k, v in timings.items() if k != "total")
    if audio_seconds <= 0 or total <= 0:
        return ""
    rtf = total / audio_seconds
    return f"{total:.0f} s para {audio_seconds:.0f} s de áudio (RTF {rtf:.2f}: {1 / rtf:.1f}x o tempo real)"


def _ctx(job: Job) -> dict:
    """Contexto/glossário do usuário como argumento opcional (só quando informado)."""
    return {"context": job.context} if getattr(job, "context", "") else {}


def _clean(job: Job, texts: list[str], spans) -> list[str]:
    """Remove alucinações e repetições (quality.py) e registra o que foi feito como avisos."""
    out, report = quality.clean_texts(texts, [max(e - s, 0.0) for s, e in spans])
    job.warnings.extend(report.messages())
    return out


def _start_warm_up(client, model: str):
    """Pré-aquecimento em segundo plano; falhas são ignoradas (a 1ª transcrição carrega o modelo de qualquer forma)."""
    fn = getattr(client, "warm_up", None)
    if not fn:
        return None

    def run():
        try:
            fn(model)
        except Exception:
            pass

    t = threading.Thread(target=run, name="warm-up", daemon=True)
    t.start()
    return t


def _transcribe_all(job: Job, client: OllamaClient, chunks, config: Config, upd):
    """Transcreve os trechos com até `config.parallel` chamadas simultâneas. Devolve (textos na ordem, falhas, último erro)."""
    n = len(chunks)
    texts: list[str] = [""] * n
    failures, done, last_error = 0, 0, ""
    workers = max(1, min(int(getattr(config, "parallel", 1) or 1), MAX_PARALLEL, n))

    def work(i: int) -> str:
        job.check_cancel()
        return srt.clean_text(client.transcribe(job.model, audio.to_wav_bytes(chunks[i].samples), **_ctx(job)))

    ex = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="transcribe")
    try:
        futures = {ex.submit(work, i): i for i in range(n)}
        for f in as_completed(futures):
            i = futures[f]
            job.check_cancel()
            done += 1
            try:
                texts[i] = f.result()
            except OllamaError as e:
                failures += 1
                last_error = str(e)
                job.warnings.append(f"Trecho {i + 1} ({srt.format_timestamp(chunks[i].start)}) falhou: {e}")
                if failures >= 3 and failures == done:  # as primeiras tentativas falham: servidor/modelo inviável
                    raise OllamaError(last_error)
            upd(15 + (TRANSCRIBE_END - 15) * done / n, f"Transcrevendo trecho {done} de {n}")
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    return texts, failures, last_error


def _batches(indexes: list[int], texts: list[str], size: int, max_chars: int = 2500) -> list[list[int]]:
    out: list[list[int]] = []
    cur: list[int] = []
    chars = 0
    for i in indexes:
        if cur and (len(cur) >= size or chars + len(texts[i]) > max_chars):
            out.append(cur)
            cur, chars = [], 0
        cur.append(i)
        chars += len(texts[i])
    if cur:
        out.append(cur)
    return out


def _needs_translation(idx: list[int], texts: list[str], target: str) -> list[int]:
    """Índices que realmente precisam de tradução: pula os textos que já estão no idioma de saída.

    Passar português por um modelo para "traduzir" para português reescreve palavras (dobrou o WER em fala real).
    Texto curto demais para identificar fica com o idioma da maioria do arquivo, se ela for o idioma de saída."""
    base = (target or "").lower().split("-")[0]
    found = {i: langid.detect(texts[i]) for i in idx}
    confident = [lang for lang in found.values() if lang]
    file_is_target = len(confident) >= 3 and sum(1 for lang in confident if lang == base) / len(confident) >= 0.7
    return [i for i in idx if not (found[i] == base or (found[i] is None and file_is_target))]


def _translate_all(job: Job, client: OllamaClient, texts: list[str], config: Config, upd, model: str | None = None) -> list[str]:
    """Traduz (ou mantém) os textos em lotes de falas consecutivas, que se dão contexto mútuo.

    Se o modelo não devolver o lote no formato esperado, cada fala é traduzida individualmente.
    """
    model = model or job.model
    idx = _needs_translation([i for i, t in enumerate(texts) if t], texts, job.language)
    if not idx:
        return texts
    batches = _batches(idx, texts, max(1, int(getattr(config, "translate_batch", 1) or 1)))
    out = list(texts)
    workers = max(1, min(int(getattr(config, "parallel", 1) or 1), MAX_PARALLEL, len(batches)))

    def work(batch: list[int]) -> list[str]:
        job.check_cancel()
        items = [texts[i] for i in batch]
        res = None
        if len(items) > 1:
            try:
                res = client.translate_batch(model, items, job.language, **_ctx(job))
            except OllamaError:
                res = None
            if res is not None and len(res) != len(items):
                res = None
        if res is None:  # lote de 1 ou resposta inválida: fala a fala
            res = []
            for i, t in zip(batch, items):
                job.check_cancel()
                try:
                    res.append(client.translate(model, t, job.language, **_ctx(job)))
                except OllamaError as e:  # mantém o texto original; segue o fluxo
                    job.warnings.append(f"Trecho {i + 1}: tradução falhou, mantido o idioma original ({e})")
                    res.append(t)
        return [srt.clean_text(r) or t for r, t in zip(res, items)]

    done = 0
    ex = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="translate")
    try:
        futures = {ex.submit(work, b): b for b in batches}
        for f in as_completed(futures):
            job.check_cancel()
            for i, t in zip(futures[f], f.result()):
                out[i] = t
            done += 1
            upd(TRANSCRIBE_END + (97 - TRANSCRIBE_END) * done / len(batches), f"Traduzindo lote {done} de {len(batches)}")
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    return out


def _ensure_engine_model(job: Job, eng: Engine, device: str = "auto") -> float:
    """Baixa o modelo do motor alternativo se ainda não estiver no disco (só quando selecionado). Devolve o deslocamento (%)."""
    server = getattr(job, "owner", None) is not None  # modo servidor: usuários não disparam downloads
    binary_share = 0.0
    if hasattr(eng, "ensure_binary") and not eng.installed(device):  # whisper.cpp/NeMo: baixa o executável (o backend depende do dispositivo)
        if server:
            raise EngineError(f"O executável do {eng.label} não está instalado neste servidor.")
        binary_share = PULL_SHARE / 2
        job.update(0, f"Baixando o executável do {eng.label}")

        def on_binary(done, total):
            job.check_cancel()
            if total:
                job.update(binary_share * done / total, f"Baixando o executável do {eng.label}",
                           f"{done / 1e6:.0f} de {total / 1e6:.0f} MB")

        eng.ensure_binary(on_binary, device)
        job.check_cancel()
    eng._require()
    if not eng.knows(job.model):
        raise EngineError(f"O modelo '{job.model}' não consta no catálogo do motor {eng.label}.")
    if eng.is_installed(job.model):
        job.message = ""
        return binary_share
    if server:
        raise EngineError("O modelo selecionado não está instalado neste servidor.")
    job.update(binary_share, f"Baixando o modelo {job.model}")

    def on_progress(done, total):
        job.check_cancel()
        if total:
            job.update(binary_share + (PULL_SHARE - binary_share) * done / total, f"Baixando o modelo {job.model}",
                       f"{done / 1e9:.2f} de {total / 1e9:.2f} GB")

    eng.ensure_model(job.model, on_progress)
    job.check_cancel()
    if not eng.is_installed(job.model):
        raise EngineError(f"O download do modelo '{job.model}' terminou, mas ele não foi encontrado no disco.")
    job.message = ""
    return PULL_SHARE


def _translation_model(client, config: Config) -> str | None:
    """Modelo do Ollama que traduz quando a transcrição veio de outro motor: o configurado, senão o 1º instalado."""
    try:
        names = [m["name"] for m in client.audio_models()]
    except OllamaError:
        return None
    pref = getattr(config, "translate_model", "")
    if pref and norm(pref) in {norm(n) for n in names}:
        return next(n for n in names if norm(n) == norm(pref))
    return names[0] if names else None


def _same_language(detected: str | None, target: str) -> bool:
    return bool(detected) and detected.lower().split("-")[0] == target.lower().split("-")[0]


def server_mode(job: Job) -> bool:
    return getattr(job, "owner", None) is not None


def _log_warnings(job: Job) -> None:
    """Os avisos do trabalho (que não aparecem mais na tela de conclusão) vão todos para o log em arquivo."""
    for w in job.warnings:
        joblog.warning("[%s] aviso: %s", job.id, w)


def run_job(job: Job, client: OllamaClient, config: Config) -> None:
    """Executa o trabalho. Os fragmentos do motor Qualidade só são apagados ao concluir ou dar erro (não ao cancelar).
    Início, fim, tempos, avisos e erros ficam no log em arquivo (applog.py)."""
    name = os.path.basename(getattr(job, "source_name", None) or job.path)
    joblog.info("[%s] início: arquivo=%s motor=%s modelo=%s idioma=%s diarização=%s contexto=%s", job.id, name, job.engine, job.model, job.language,
                job.diarize, "sim" if getattr(job, "context", "") else "não")
    try:
        _run_job(job, client, config)
    except Cancelled:
        joblog.info("[%s] cancelado na etapa '%s' (%.0f%%)", job.id, job.stage, job.progress)
        _log_warnings(job)
        raise  # regra do usuário: fragmentos ficam; workdir.purge_old limpa os antigos
    except BaseException as e:
        joblog.error("[%s] erro: %s", job.id, e, exc_info=not isinstance(e, (EngineError, OllamaError, MediaError)))
        _log_warnings(job)
        workdir.remove(job.work_dir)
        raise
    else:
        joblog.info("[%s] concluído: %s · áudio=%.0f s · tempos=%s · saída=%s", job.id, job.message, job.audio_seconds, job.timings, job.output_path)
        _log_warnings(job)
        workdir.remove(job.work_dir)


def _run_job(job: Job, client: OllamaClient, config: Config) -> None:
    clock = _Stopwatch(job.timings)
    eng = None
    config = quality_engine.apply_mode(job.engine, config)  # modos de qualidade: revisor, B, refinamento, especialistas fixos
    fixed = quality_engine.mode_generator(job.engine)
    if fixed:
        job.model = fixed  # modos de qualidade têm o gerador fixo: o modelo da tela é ignorado
    if job.engine != engines.OLLAMA and not quality_engine.is_quality(job.engine):
        eng = engines.get(job.engine)
        if not eng:
            raise EngineError(f"Motor de transcrição desconhecido: {job.engine}")
        if job.diarize and not eng.supports_diarization:
            raise EngineError(f"O motor {eng.label} não faz diarização. Use o WhisperX.")
        lo = _ensure_engine_model(job, eng, getattr(config, "device", "auto"))
    else:
        lo = _ensure_model(job, client)
        if fixed and not server_mode(job):  # o revisor fixo também precisa estar instalado (no servidor nada é baixado)
            lo = max(lo, _ensure_model(job, client, config.quality_reviewer))

    clock.lap("modelo")  # verificação/download do modelo

    def upd(p, stage=None, message=None):  # mapeia 0–100 para [lo, 100]
        job.update(lo + p * (100 - lo) / 100, stage, message)

    upd(0, "Validando arquivo")
    info = media.probe(job.path)
    job.check_cancel()
    limit = getattr(config, "max_duration_min", 0)
    if limit and info.duration > limit * 60:
        raise MediaError(f"A mídia tem {info.duration / 60:.0f} min; o limite deste servidor é {limit:.0f} min.")
    upd(5, "Extraindo áudio do vídeo" if info.kind == "video" else "Preparando áudio")
    warm = None if eng else _start_warm_up(client, job.model)  # carrega o modelo na GPU enquanto o ffmpeg extrai o áudio

    with tempfile.TemporaryDirectory(prefix="minitranscritor_") as tmp:
        wav = os.path.join(tmp, "audio.wav")
        audio.extract_wav(job.path, wav)
        job.check_cancel()
        samples = audio.read_wav(wav)
    clock.lap("extracao")

    upd(15, "Dividindo em trechos")
    if getattr(config, "vad", True):  # corta nas pausas e pula o silêncio
        chunks = audio.split_by_speech(samples, config.chunk_seconds, config.max_chunk_seconds)
    else:
        chunks = audio.split_chunks(samples, config.chunk_seconds, config.max_chunk_seconds)
    if not chunks:
        raise MediaError("Nenhuma fala detectada no áudio (arquivo silencioso).")
    clock.lap("vad")

    if eng:
        def eng_upd(p, stage=None):
            upd(15 + (TRANSCRIBE_END - 15) * p / 100, stage)

        segs, detected = eng.transcribe(job.model, samples, chunks, job, eng_upd, diarize=job.diarize,
                                        hf_token=getattr(config, "hf_token", ""), parallel=getattr(config, "parallel", 1),
                                        context=getattr(job, "context", ""), device=getattr(config, "device", "auto"))
        clock.lap("transcricao")
        if not segs:
            raise MediaError("O modelo não retornou texto para nenhum trecho.")
        spans = [(sg.start, sg.end) for sg in segs]
        texts = _clean(job, [sg.text for sg in segs], spans)
        job.source_cues = [(s, e, t) for (s, e), t in zip(spans, texts) if t]
        if not _same_language(detected, job.language):  # já está no idioma de saída? então não traduz
            tmodel = _translation_model(client, config)
            if tmodel:
                texts = _translate_all(job, client, texts, config, upd, model=tmodel)
            else:
                job.warnings.append(f"Sem modelo no Ollama para traduzir: a legenda ficou no idioma original ({detected or 'n/d'}).")
        speakers = [sg.speaker for sg in segs] if job.diarize else [None] * len(segs)
        clock.lap("traducao")
    else:
        if warm:
            warm.join(timeout=config.ollama_timeout)
        if quality_engine.is_quality(job.engine):  # 2 gerações + portão + revisor + juiz (ver docs/proposta-multiagente.md)
            texts, failures, last_error, report = quality_engine.run(job, client, config, samples, chunks, upd, _transcribe_all)
            job.report = report.as_dict()
        else:
            texts, failures, last_error = _transcribe_all(job, client, chunks, config, upd)
        clock.lap("transcricao")
        if failures == len(chunks):
            raise OllamaError(last_error or "Todos os trechos falharam.")
        spans = [(ch.start, ch.end) for ch in chunks]
        texts = _clean(job, texts, spans)
        job.source_cues = [(s, e, t) for (s, e), t in zip(spans, texts) if t]
        src_texts = list(texts)
        texts = _translate_all(job, client, texts, config, upd, model=((getattr(config, "translate_model", "") or None) if quality_engine.is_quality(job.engine) else None))
        if quality_engine.is_quality(job.engine):  # especialista de tradução: a legenda só é montada depois dele
            texts = quality_engine.review_translation_step(job, client, config, src_texts, texts, upd)
        speakers = [None] * len(chunks)
        clock.lap("traducao")

    labels = {}
    for sp in speakers:
        if sp and sp not in labels:
            labels[sp] = f"Falante {len(labels) + 1}"  # SPEAKER_00 → "Falante 1", na ordem de aparição
    cues: list[srt.Cue] = []
    prev = None
    for (start, end), text, sp in zip(spans, texts, speakers):
        if not text:
            continue
        if sp and sp != prev:
            text = f"[{labels[sp]}] {text}"
        prev = sp or prev
        cues.extend(srt.cues_for_chunk(start, end, text))
    if not cues:
        raise MediaError("O modelo não retornou texto para nenhum trecho.")

    cues = srt.apply_style(cues)  # linhas de até 42 caracteres, tempos de leitura e intervalos
    fast = srt.fast_reading(cues)
    if fast:
        job.warnings.append(f"{fast} legenda(s) passam de {srt.MAX_CPS:.0f} caracteres por segundo (leitura rápida): o trecho tem pouco tempo para o texto.")
    upd(97, "Gravando legenda")
    job.output_path = _write_srt(job, srt.to_srt(cues), config)
    clock.lap("montagem")
    job.timings["total"] = round(sum(job.timings.values()), 2)
    job.audio_seconds = info.duration
    job.message = f"Tempo de processamento: {format_hms(job.timings['total'])}"  # detalhes (RTF, etapas) ficam em job.timings/audio_seconds
