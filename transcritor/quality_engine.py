"""Motor "Qualidade": 2 gerações independentes + portão de divergência + revisor com áudio + juiz, repetidos em passadas de
refinamento, e por fim os especialistas (a legenda só é montada depois deles).

Desenho completo em docs/proposta-multiagente.md. Aqui está a lógica do pipeline como funções puras/testáveis, sem
depender do ADK (a orquestração pelo ADK é uma camada fina por cima, prevista na proposta). Fases, para GPU única:

  1. Geradores — A (modelo de áudio do Ollama = o da tela) e B (motor Whisper; no modo padrão, o WhisperX) transcrevem TODOS os
                 fragmentos, um gerador por vez
  2. Portão    — código: fragmentos em que as gerações divergem (WER) ou em que só algumas têm texto
  3. Revisor   — modelo de áudio do Ollama REOUVE só os fragmentos marcados e devolve JSON (veredito + texto + confiança e, quando
                 há refinamento, a crítica e os termos que ouviu)
  4. Juiz      — código, com travas; na dúvida vale a geração prioritária (ou a mais central, com 3 candidatos)
  5. Refinamento (`quality_loops`, 2 nos modos quality-e4b/-12b = 3 passadas) — os geradores transcrevem de novo os fragmentos
                 ainda em dúvida recebendo a versão final anterior + a crítica do revisor (A: no prompt com o áudio; B/WhisperX:
                 como initial_prompt); a versão anterior entra como candidato C; portão → revisor → juiz de novo
  6. Especialistas (`quality_specialists`) — termos, pontuação (aqui) e revisão da tradução (depois da tradução, no pipeline)

Os fragmentos ficam numa pasta de trabalho (`work/<id>/`, ver workdir.py), apagada só quando o trabalho conclui ou dá erro.
"""
import logging
import bisect
import dataclasses
import time
from dataclasses import dataclass, field

from . import audio, engines, judge_v2, metrics, prompts, quality, specialists, speechgate, srt, vote, workdir
from .catalog import norm
from .ollama_client import OllamaError, build_feedback_note

log = logging.getLogger(__name__)

QUALITY = "quality"
# Modos de qualidade do app, com modelos FIXOS (o modelo escolhido na tela é ignorado). Cada modo é a melhor configuração medida
# em docs/experimento-loop-refinamento.md; a escolha é do projeto, não do usuário.
#  * quality-1: autorrefinamento gerador <-> revisor SÓ com modelos do Ollama (sem WhisperX), com porteiro de fala (Silero) antes do
#    gerador e 2 passadas. Melhor configuração medida: gerador e4b + revisor 12b + porteiro, sem votação (filme: 46,3 -> 25,5 % de WER,
#    inserções 22,9 -> 3,7 %, 18 min; antes do porteiro, 35,0 %). Ainda longe do WhisperX (11,5 %).
#  * quality-2: WhisperX (B) + gerador A + revisor com áudio, v2.1 (âncora no B, silêncio, confiança do A, decisão por ponto),
#    2 passadas e especialistas. Melhor combinação nos clipes curtos (4,11 % contra 5,41 % do WhisperX sozinho, nenhum clipe pior)
#    e empate no filme (11,6 % contra 11,5 %).
MODES = {
    "quality-1": {"label": "Qualidade 1 — Autorrefinamento (e4b → 12b)", "generator": "gemma4:e4b", "reviewer": "gemma4:12b",
                  "self": True, "loops": 1, "specialists": False, "gate": True,
                  "description": "Só modelos do Ollama, sem WhisperX: um detector de voz descarta o que não é fala, o gemma4:e4b transcreve e o gemma4:12b "
                                 "reouve cada trecho, aponta o que está errado e o gerador refaz, 1 vez. Modelos fixos. Corta texto inventado, mas erra mais que o WhisperX."},
    "quality-2": {"label": "Qualidade 2 — WhisperX + Revisão (e4b → 12b)", "generator": "gemma4:e4b", "reviewer": "gemma4:12b",
                  "self": False, "loops": 1, "specialists": True, "gate": False,
                  "description": "WhisperX e o gemma4:e4b transcrevem; o gemma4:12b reouve só os trechos em dúvida, com 1 refinamento e especialistas. "
                                 "Modelos fixos. Mais lento (~2x o WhisperX); mede igual ou melhor que o WhisperX sozinho."},
}
MODE_B_ENGINE, MODE_B_MODEL = "whisperx", "large-v3-turbo"
DEFAULT_B_ENGINE = "openai-whisper"
DEFAULT_B_MODEL = "large-v3-turbo"
FALLBACK_B_MODEL = "gemma4:e2b"  # 2ª geração do Ollama, se o motor Whisper não estiver instalado
THRESHOLD = 0.05  # WER entre gerações acima do qual o fragmento vai à revisão (0,10 deixava 71 % dos erros de fora; 0,05 = melhor custo/benefício, 2026-11-11)
MIN_CONFIDENCE = 0.5
MAX_MERGE_WER = 0.30  # o texto "mesclado" do revisor pode se afastar até isto da melhor entre as gerações
LABELS = "ABC"


def is_quality(engine_id: str) -> bool:
    return engine_id == QUALITY or engine_id in MODES


def apply_mode(engine_id: str, config):
    """Config efetiva do modo (tudo fixo): o 1 = autorrefinamento sem WhisperX; o 2 = WhisperX + v2.1 com especialistas."""
    mode = MODES.get(engine_id)
    if not mode:
        return config
    common = dict(quality_reviewer=mode["reviewer"], quality_loops=mode["loops"], quality_specialists=mode["specialists"], quality_prefer="B")
    if mode["self"]:
        return dataclasses.replace(config, quality_self=True, quality_judge="v1", quality_gate=mode["gate"], quality_vote=False, **common)
    return dataclasses.replace(config, quality_b_engine=MODE_B_ENGINE, quality_b_model=MODE_B_MODEL, quality_judge="v2",
                               quality_feedback="terms", quality_triage=False, quality_self=False, **common)


def mode_generator(engine_id: str) -> str:
    """Modelo do gerador A fixo do modo (vazio fora dos modos fixos: vale o modelo escolhido na tela)."""
    return (MODES.get(engine_id) or {}).get("generator", "")


@dataclass
class Fragment:
    index: int
    start: float
    end: float
    a: str = ""
    b: str = ""
    c: str = ""
    flagged: bool = False
    review: dict | None = None
    final: str = ""
    source: str = "A"  # A | B | C | merge | central | X(...)  (de onde veio o texto final)
    passes: list = field(default_factory=list)  # por passada: {"pass","a","b","c","flagged","review","final","source"}
    pre_specialists: str = ""  # texto final antes dos especialistas (vazio = não mudou)


@dataclass
class QualityReport:
    fragments: int = 0
    flagged: int = 0
    reviewed: int = 0
    from_a: int = 0
    from_b: int = 0
    from_c: int = 0
    merged: int = 0
    kept_a: int = 0  # revisão rejeitada pelas travas ou sem confiança (valeu a geração prioritária/central)
    b_source: str = ""
    reviewer: str = ""
    loops: int = 0  # passadas de refinamento pedidas
    loops_done: int = 0  # realmente executadas (parou antes se nada mudou)
    passes: list = field(default_factory=list)  # por passada: {"pass","targets","flagged","reviewed","changed","seconds"}
    specialists: dict = field(default_factory=dict)
    judge: str = "v1"  # v1 | v2
    silenced: int = 0  # v2: fragmentos em que o B calou e o A foi ignorado (silêncio)
    signals: dict = field(default_factory=dict)  # v2: confiança do A medida no arquivo (razão A/B, fração só-A, a_trusted)
    triage: dict = field(default_factory=dict)  # v2: chamadas, suspeitos e termos da triagem do documento
    gate: dict = field(default_factory=dict)  # modo 1: porteiro de fala (fragmentos descartados antes de gerar)
    vote: dict = field(default_factory=dict)  # modo 1: votação entre hipóteses (trocas por maioria, dúvidas, instáveis)
    sources: list = field(default_factory=list)
    seconds: dict = field(default_factory=dict)
    texts: dict = field(default_factory=dict)  # {"pass1": [...], "pass2": [...], "pre_specialists": [...], "final": [...]} (não vai na API)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d.pop("texts", None)
        return d


def _lang(language: str) -> str:
    return (language or "").lower().split("-")[0]


def gate_multi(gens: list[list[str]], language: str, threshold: float = THRESHOLD) -> list[int]:
    """Índices dos fragmentos que vão ao revisor: algum par de gerações diverge mais que o limiar, ou só algumas têm texto."""
    lang = _lang(language)
    flagged = []
    for i in range(len(gens[0]) if gens else 0):
        cands = [g[i] if i < len(g) else "" for g in gens]
        filled = [t for t in cands if t]
        if not filled:
            continue
        if len(filled) < len(cands):
            flagged.append(i)
            continue
        if any(metrics.wer(x, y, lang) > threshold for n, x in enumerate(filled) for y in filled[n + 1:]):
            flagged.append(i)
    return flagged


def gate(a: list[str], b: list[str], language: str, threshold: float = THRESHOLD) -> list[int]:
    return gate_multi([a, b], language, threshold)


def central(cands: list[str], lang: str) -> int:
    """Índice da geração mais central (menor soma de WER para as demais): o voto da maioria entre 3 gerações."""
    best, best_i = None, 0
    for i, x in enumerate(cands):
        if not x:
            continue
        total = sum(metrics.wer(x, y, lang) for j, y in enumerate(cands) if j != i and y)
        if best is None or total < best:
            best, best_i = total, i
    return best_i


def judge(a: str, b: str, review: dict | None, language: str, glossary: str = "",
          min_confidence: float = MIN_CONFIDENCE, prefer: str = "A", *, c: str = "") -> tuple[str, str]:
    """Aplica o veredito do revisor com travas. Devolve (texto final, origem).

    Sem revisão ou com qualquer trava falhando, não se muda: vale a geração prioritária (`prefer`: A ou B) ou, havendo uma
    terceira geração (`c`), a mais central entre as três."""
    lang = _lang(language)
    cands = {"A": a, "B": b, "C": c}
    if c:
        pref_text, pref_label = cands[LABELS[central([a, b, c], lang)]], "central"
    else:
        use_b = str(prefer).upper() == "B" and bool(b)
        pref_label, pref_text = ("B", b) if use_b else ("A", a)
    if not review:
        return pref_text, pref_label
    verdict = str(review.get("verdict") or "").strip().upper()
    try:
        conf = float(review.get("confidence") or 0)
    except (TypeError, ValueError):
        conf = 0.0
    text = srt.clean_text(str(review.get("text") or ""))
    if conf < min_confidence:
        return pref_text, f"{pref_label}(confiança baixa)"
    if verdict in ("A", "B", "C"):
        chosen = cands[verdict]
        return (chosen, verdict) if chosen else (pref_text, f"{pref_label}({verdict} vazio)")
    if verdict in ("MERGE", "MESCLA") and text:
        pool = [t for t in (a, b, c) if t]
        allowed = set(metrics.normalize(glossary, lang))
        for t in pool:
            allowed |= set(metrics.normalize(t, lang))
        words = metrics.normalize(text, lang)
        close = min((metrics.wer(t, text, lang) for t in pool), default=1.0)
        if words and all(w in allowed for w in words) and close <= MAX_MERGE_WER:
            return text, "merge"
        return pref_text, f"{pref_label}(mescla rejeitada)"
    return pref_text, f"{pref_label}(veredito inválido)"


def group_by_fragment(segs, chunks) -> list[str]:
    """Distribui segmentos com timestamps absolutos (de um motor Whisper) pelos fragmentos, pelo ponto médio."""
    starts = [c.start for c in chunks]
    out: list[list[str]] = [[] for _ in chunks]
    for s in segs:
        mid = (s.start + s.end) / 2
        i = max(0, min(bisect.bisect_right(starts, mid) - 1, len(chunks) - 1))
        out[i].append(s.text)
    return [" ".join(parts).strip() for parts in out]


class _Shadow:
    """Job com outro modelo, só para reaproveitar _transcribe_all (que lê job.model, check_cancel, warnings, context)."""

    def __init__(self, job, model):
        self._job, self.model = job, model

    def __getattr__(self, name):
        return getattr(self._job, name)


def _open_session(job, client, config, engine_id: str, model: str):
    """Sessão (modelo carregado, prompt por fragmento) do gerador B, se o motor tiver uma e o modelo estiver utilizável; senão None."""
    eng = engines.get(engine_id)
    if not (eng and eng.available() and eng.knows(model)) or not hasattr(eng, "session"):
        return None, ""
    if not eng.is_installed(model):
        if getattr(job, "owner", None) is not None:
            raise engines.EngineError("O modelo desta geração não está instalado neste servidor.")
        eng.ensure_model(model)
    dev = getattr(config, "device", "auto")
    if hasattr(client, "unload_all") and engines.pick_device(dev, ("cpu", "cuda"), "")[0] == "cuda":
        client.unload_all()  # com o reconhecimento na GPU (CUDA), libera a VRAM do Ollama; na CPU não é preciso
    sess = eng.session(model, device=dev, context=getattr(job, "context", ""), job=job)
    return sess, f"{eng.label} {model}" if sess else ""


def _generate_b(job, sess, chunks, prompts, upd, label: str) -> list[str]:
    out = []
    for i, ch in enumerate(chunks):
        job.check_cancel()
        upd(100 * i / max(len(chunks), 1), f"{label}: fragmento {i + 1} de {len(chunks)}")
        out.append(quality.collapse_repeats(srt.clean_text(sess.transcribe(ch.samples, prompts[i] if prompts else ""))))
    return out


def _generate(job, client, config, samples, chunks, upd, engine_id: str, model: str) -> tuple[list[str], str]:
    """Uma geração em TODOS os fragmentos: `ollama` (modelo de áudio do Ollama) ou um motor Whisper. Devolve (textos, descrição)."""
    from . import pipeline  # import tardio: evita ciclo

    if engine_id == engines.OLLAMA:
        texts, _f, _e = pipeline._transcribe_all(_Shadow(job, model), client, chunks, config, lambda p, stage=None: upd(p, stage))
        return texts, f"Ollama {model}"
    eng = engines.get(engine_id)
    if eng and eng.available() and eng.knows(model):
        if hasattr(client, "unload_all"):
            client.unload_all()  # o Whisper na GPU precisa de VRAM livre (os modelos do Ollama recarregam sob demanda)
        if not eng.is_installed(model):
            if getattr(job, "owner", None) is not None:
                raise engines.EngineError("O modelo desta geração não está instalado neste servidor.")
            eng.ensure_model(model)
        segs, _lang = eng.transcribe(model, samples, chunks, job, lambda p, stage=None: upd(p, stage),
                                     context=getattr(job, "context", ""), device=getattr(config, "device", "auto"))
        engines.release_memory()
        return [quality.collapse_repeats(t) for t in group_by_fragment(segs, chunks)], f"{eng.label} {model}"
    job.warnings.append(f"Motor {engine_id} indisponível para a 2ª geração: usando o Ollama {FALLBACK_B_MODEL}.")
    texts, _f, _e = pipeline._transcribe_all(_Shadow(job, FALLBACK_B_MODEL), client, chunks, config, lambda p, stage=None: upd(p, stage))
    return texts, f"Ollama {FALLBACK_B_MODEL}"


def _review_one(job, client, reviewer, chunk, a, b, c, notes, c_prev, ctx, n_label) -> dict | None:
    kw = {}
    if c:
        kw["c"] = c
    if notes:
        kw["notes"] = True
    if c and c_prev:
        kw["c_is_previous"] = True
    try:
        return client.review(reviewer, audio.to_wav_bytes(chunk.samples), a, b, context=ctx, **kw)
    except OllamaError as e:
        _review_failed(job, n_label, e, "mantida a geração prioritária")
        return None


def _review_failed(job, fragment, error, kept: str) -> None:
    """Revisão de um fragmento que falhou: o detalhe vai para o log e o job guarda UMA linha de aviso com a contagem (antes eram dezenas
    de linhas, uma por fragmento, que só atrapalhavam a leitura)."""
    log.warning("Revisão do fragmento %s falhou (%s); %s.", fragment, error, kept)
    st = getattr(job, "_review_fail", None)
    if st is None:
        st = {"n": 0, "idx": None}
        job._review_fail = st
    st["n"] += 1
    msg = f"{st['n']} revisão(ões) de fragmento sem resposta válida do revisor; {kept}."
    if st["idx"] is None or st["idx"] >= len(job.warnings):
        job.warnings.append(msg)
        st["idx"] = len(job.warnings) - 1
    else:
        job.warnings[st["idx"]] = msg


def _feedback_for(f: Fragment, glossary: list[str]) -> tuple[str, str]:
    """(nota para o gerador A, prompt para o B): versão final anterior + crítica e termos do revisor."""
    r = f.review or {}
    critique = str(r.get("critique") or "").strip()
    terms = [t for t in (r.get("terms") or []) if isinstance(t, str)] or glossary[:8]
    return (build_feedback_note(f.final, critique, terms),
            (prompts.render("whisper_terms", terms=", ".join(terms[:8])) + " " if terms else "") + f.final)


def run(job, client, config, samples, chunks, upd, transcribe_a) -> tuple[list[str], int, str, QualityReport]:
    """Executa o pipeline. `transcribe_a(job, client, chunks, config, upd)` é o `_transcribe_all` do pipeline (geração A).

    Devolve (texto final por fragmento, falhas de A, último erro, relatório)."""
    report = QualityReport(fragments=len(chunks))
    language = job.language
    ctx = getattr(job, "context", "")
    wd = workdir.create(config, job.id)
    job.work_dir = wd
    workdir.save_fragments(wd, chunks)
    loops = max(0, int(getattr(config, "quality_loops", 0) or 0))
    report.loops = loops
    n = len(chunks)

    # --- barra de progresso: gerações 0–45 %, 1ª revisão 45–60 %, cada refinamento divide 60–92 %, especialistas 92–95 %
    gen_end = 45.0 if loops else 75.0
    rev_end = 60.0 if loops else 95.0
    other = [(getattr(config, "quality_b_engine", DEFAULT_B_ENGINE), getattr(config, "quality_b_model", DEFAULT_B_MODEL))]
    if getattr(config, "quality_c_engine", ""):
        other.append((config.quality_c_engine, getattr(config, "quality_c_model", "") or DEFAULT_B_MODEL))
    share = gen_end / (1 + len(other))  # fatia da barra de cada geração

    t0 = time.perf_counter()
    keep = list(range(n))
    if getattr(config, "quality_gate", False) and speechgate.available():  # porteiro: fragmentos sem fala nem chegam ao gerador
        ratios = [speechgate.speech_ratio(ch.samples) for ch in chunks]
        keep = [i for i, r in enumerate(ratios) if r > speechgate.MIN_RATIO]
        if not keep:  # o detector não achou fala em NADA: provável áudio fora do que ele conhece; melhor transcrever tudo do que entregar vazio
            job.warnings.append("O porteiro de fala não achou fala em nenhum fragmento: todos foram transcritos.")
            keep = list(range(n))
        report.gate = {"fragments": n, "discarded": n - len(keep), "seconds": round(time.perf_counter() - t0, 1)}
    elif getattr(config, "quality_gate", False):
        job.warnings.append("Porteiro de fala indisponível (faster-whisper/onnxruntime não instalados): todos os fragmentos foram transcritos.")
    if len(keep) == n:
        texts_a, failures, last_error = transcribe_a(job, client, chunks, config, lambda p, stage=None: upd(p * share / 100, stage))
    elif keep:
        sub, failures, last_error = transcribe_a(job, client, [chunks[i] for i in keep], config, lambda p, stage=None: upd(p * share / 100, stage))
        texts_a = [""] * n
        for k, i in enumerate(keep):
            texts_a[i] = sub[k]
    else:
        texts_a, failures, last_error = [""] * n, 0, ""
    report.seconds["geracao_a"] = round(time.perf_counter() - t0, 1)
    report.sources = [f"Ollama {job.model}"]
    if keep and failures == len(keep):
        return texts_a, failures, last_error, report

    if getattr(config, "quality_self", False):  # modo 1: autorrefinamento gerador <-> revisor, sem gerador B
        return _after_generation_self(job, client, config, chunks, upd, texts_a, report, language, ctx, loops, wd, failures, last_error)
    gens = [texts_a]
    session = None
    for k, (eid, model) in enumerate(other, 1):
        job.check_cancel()
        t0 = time.perf_counter()
        base = share * k
        cb = lambda p, stage=None, base=base, k=k: upd(base + p * share / 100, stage or f"{LABELS[k]}ª geração")  # noqa: E731
        sess, desc = (_open_session(job, client, config, eid, model) if k == 1 and eid != engines.OLLAMA else (None, ""))
        if sess:
            try:
                texts = _generate_b(job, sess, chunks, None, cb, f"{LABELS[k]}ª geração ({desc})")
            except BaseException:
                sess.close()
                raise
            if loops:
                session = sess  # fica aberta para os refinamentos
            else:
                sess.close()
        else:
            texts, desc = _generate(job, client, config, samples, chunks, cb, eid, model)
        report.seconds[f"geracao_{LABELS[k].lower()}"] = round(time.perf_counter() - t0, 1)
        report.sources.append(desc)
        gens.append((texts + [""] * n)[:n])
    report.b_source = report.sources[1]

    try:
        return _after_generation(job, client, config, chunks, upd, gens, report, session, language, ctx, loops, wd, failures,
                                 last_error, rev_end)
    finally:
        if session:
            session.close()


def _after_generation(job, client, config, chunks, upd, gens, report, session, language, ctx, loops, wd, failures, last_error, rev_end):
    if getattr(config, "quality_judge", "v1") == "v2":
        return _after_generation_v2(job, client, config, chunks, upd, gens, report, session, language, ctx, loops, wd, failures,
                                    last_error, rev_end)
    n = len(chunks)
    texts_a, texts_b = gens[0], gens[1]
    texts_c = gens[2] if len(gens) > 2 else [""] * n
    prefer = str(getattr(config, "quality_prefer", "A")).upper()  # geração prioritária quando não há dúvida/revisão
    min_conf = getattr(config, "quality_min_conf", MIN_CONFIDENCE)
    threshold = getattr(config, "quality_threshold", THRESHOLD)
    frags = []
    for i, ch in enumerate(chunks):
        a, b, c = texts_a[i], texts_b[i], texts_c[i]
        if len(gens) > 2:
            final, src = judge(a, b, None, language, "", prefer=prefer, c=c)
        else:
            final, src = (b, "B") if prefer == "B" and b else (a, "A")
        frags.append(Fragment(i, ch.start, ch.end, a=a, b=b, c=c, final=final, source=src))
    flagged = gate_multi(gens, language, threshold)
    for i in flagged:
        frags[i].flagged = True
    report.flagged = len(flagged)

    reviewer = _resolve_reviewer(job, client, config, report)

    # ---------------------------------------------------------------- passada 1: revisão dos fragmentos marcados
    t0 = time.perf_counter()
    all_reviews: list[dict] = []
    for k, i in enumerate(flagged, 1):
        job.check_cancel()
        upd(rev_end - 15 + 15 * (k - 1) / max(len(flagged), 1) if loops else 75 + 20 * (k - 1) / max(len(flagged), 1),
            f"Revisando fragmento {i + 1} ({k} de {len(flagged)})")
        f = frags[i]
        f.review = _review_one(job, client, reviewer, chunks[i], f.a, f.b, f.c if len(gens) > 2 else "", loops > 0, False, ctx, i + 1)
        report.reviewed += 1
        if f.review:
            all_reviews.append(f.review)
        f.final, f.source = judge(f.a, f.b, f.review, language, ctx, min_conf, prefer, c=f.c)
        _count(report, f.source)
    report.seconds["revisao"] = round(time.perf_counter() - t0, 1)
    for f in frags:
        f.passes.append({"pass": 1, "a": f.a, "b": f.b, "c": f.c, "flagged": f.flagged, "review": f.review, "final": f.final, "source": f.source})
    report.passes.append({"pass": 1, "targets": n, "flagged": len(flagged), "reviewed": len(flagged),
                          "changed": sum(1 for f in frags if f.final != (f.b if prefer == "B" and f.b else f.a)),
                          "seconds": report.seconds["revisao"]})
    report.texts["pass1"] = [f.final for f in frags]
    report.texts["a1"], report.texts["b1"] = list(texts_a), list(texts_b)

    # ---------------------------------------------------------------- passadas 2..: refinamento com feedback
    targets = list(flagged)
    glossary = specialists.build_glossary(ctx, all_reviews)
    for it in range(1, loops + 1):
        if not targets:
            break
        job.check_cancel()
        last = it == loops
        lo = rev_end + (92 - rev_end) * (it - 1) / loops
        span = (92 - rev_end) / loops
        t0 = time.perf_counter()
        notes_next = not last
        before = {i: frags[i].final for i in targets}
        new_a, new_b = {}, {}
        # A: Ollama, com o áudio + versão anterior + crítica (um modelo por vez: depois vêm o B e o revisor)
        for k, i in enumerate(targets, 1):
            job.check_cancel()
            upd(lo + span * 0.4 * (k - 1) / len(targets), f"Passada {it + 1}: gerador A, fragmento {i + 1} ({k} de {len(targets)})")
            fa, _ = _feedback_for(frags[i], glossary)
            try:
                new_a[i] = srt.clean_text(client.transcribe(job.model, audio.to_wav_bytes(chunks[i].samples), **({"context": ctx} if ctx else {}), feedback=fa))
            except OllamaError as e:
                job.warnings.append(f"Passada {it + 1}: gerador A falhou no fragmento {i + 1} ({e}); mantida a geração anterior.")
                new_a[i] = frags[i].passes[-1]["a"]
        # B: WhisperX (ou o que houver), com initial_prompt; sem sessão, B continua igual
        for k, i in enumerate(targets, 1):
            job.check_cancel()
            upd(lo + span * (0.4 + 0.2 * (k - 1) / len(targets)), f"Passada {it + 1}: gerador B, fragmento {i + 1} ({k} de {len(targets)})")
            if session:
                _, pb = _feedback_for(frags[i], glossary)
                new_b[i] = quality.collapse_repeats(srt.clean_text(session.transcribe(chunks[i].samples, pb)))
            else:
                new_b[i] = frags[i].passes[-1]["b"]
        # portão + revisor + juiz, com a versão anterior como candidato C
        next_targets = []
        reviewed = 0
        for k, i in enumerate(targets, 1):
            job.check_cancel()
            f = frags[i]
            a, b = new_a[i], new_b[i]
            prev = f.final
            c = prev if prev and prev != a and prev != b else ""
            gens_i = [[a], [b]] + ([[c]] if c else [])
            flagged_now = bool(gate_multi(gens_i, language, threshold))
            review = None
            if flagged_now:
                upd(lo + span * (0.6 + 0.4 * (k - 1) / len(targets)), f"Passada {it + 1}: revisando fragmento {i + 1} ({k} de {len(targets)})")
                review = _review_one(job, client, reviewer, chunks[i], a, b, c, notes_next, True, ctx, i + 1)
                reviewed += 1
                if review:
                    all_reviews.append(review)
            f.review = review if flagged_now else f.review
            f.final, f.source = judge(a, b, review, language, ctx, min_conf, prefer, c=c)
            f.passes.append({"pass": it + 1, "a": a, "b": b, "c": c, "flagged": flagged_now, "review": review, "final": f.final, "source": f.source})
            if flagged_now:
                next_targets.append(i)
        changed = sum(1 for i in targets if frags[i].final != before[i])
        secs = round(time.perf_counter() - t0, 1)
        report.passes.append({"pass": it + 1, "targets": len(targets), "flagged": len(next_targets), "reviewed": reviewed,
                              "changed": changed, "seconds": secs})
        report.seconds[f"refino_{it}"] = secs
        report.loops_done = it
        report.texts[f"pass{it + 1}"] = [f.final for f in frags]
        targets = next_targets
        if changed == 0:  # nada mudou: mais uma passada daria o mesmo
            break
    glossary = specialists.build_glossary(ctx, all_reviews)
    return _finish(job, client, config, frags, report, reviewer, glossary, wd, failures, last_error, upd)


def _resolve_reviewer(job, client, config, report) -> str:
    reviewer = getattr(config, "quality_reviewer", "") or job.model
    try:  # o revisor padrão é o maior modelo (gemma4:12b); se não estiver instalado, usa o modelo da geração A
        installed = {norm(m["name"]) for m in client.audio_models()}
        if norm(reviewer) not in installed:
            job.warnings.append(f"Revisor {reviewer} não está instalado: usando {job.model} (a revisão fica menos precisa).")
            reviewer = job.model
    except OllamaError:
        pass
    report.reviewer = reviewer
    return reviewer


def _finish(job, client, config, frags, report, reviewer, glossary, wd, failures, last_error, upd):
    """Especialistas (antes de montar a legenda) e manifesto; comum à v1 e à v2."""
    if getattr(config, "quality_specialists", False):
        t0 = time.perf_counter()
        upd(92, "Especialistas: termos e pontuação")
        finals = [f.final for f in frags]
        report.texts["pre_specialists"] = list(finals)
        new, n_terms = specialists.apply_terms(finals, glossary)
        report.specialists = {"terms_glossary": len(glossary), "terms_changed": n_terms}
        try:
            new, pstats = specialists.polish(client, reviewer, new, "", ", ".join(glossary[:30]), check=job.check_cancel)
        except OllamaError:
            pstats = {"changed": 0, "rejected": 0, "failed_batches": -1}
        report.specialists["polish"] = pstats
        for f, t in zip(frags, new):
            if t != f.final:
                f.pre_specialists, f.final = f.final, t
        report.seconds["especialistas"] = round(time.perf_counter() - t0, 1)
    report.specialists["glossary"] = glossary[:40]
    report.texts["final"] = [f.final for f in frags]

    workdir.save_manifest(wd, frags, report)
    return [f.final for f in frags], failures, last_error, report


def _neighbours(frags, i: int, k: int) -> tuple[str, str]:
    """Texto dos k fragmentos antes e depois (só como contexto do revisor; nunca vai a um gerador)."""
    before = " ".join(f.final for f in frags[max(0, i - k):i] if f.final)[-240:] if k > 0 else ""
    after = " ".join(f.final for f in frags[i + 1:i + 1 + k] if f.final)[:240] if k > 0 else ""
    return before, after


def _review_v2(job, client, config, reviewer, frags, chunks, i, a, b, terms, notes, trusted=False) -> dict | None:
    diffs = judge_v2.word_diffs(a, b) if (a and b) else []
    before, after = _neighbours(frags, i, int(getattr(config, "quality_neighbours", 2) or 0))
    ctx = ", ".join(t for t in [getattr(job, "context", "")] + list(terms[:30]) if t)
    try:
        return client.review_v2(reviewer, audio.to_wav_bytes(chunks[i].samples), a, b,
                                None if len(diffs) > judge_v2.MAX_DIFFS else diffs, chunks[i].end - chunks[i].start,
                                context=ctx, before=before, after=after, notes=notes, **({"trusted": True} if trusted else {}))
    except OllamaError as e:
        _review_failed(job, i + 1, e, "mantido o texto do WhisperX")
        return None


def _count_v2(report: QualityReport, src: str) -> None:
    if src.startswith("silêncio"):
        report.silenced += 1
    elif src.startswith("A+B"):
        report.merged += 1
    elif src.startswith("A("):
        report.from_a += 1
    else:
        report.from_b += 1


def _triage(job, client, config, reviewer, texts_a, texts_b, report) -> tuple[set, list]:
    """Triagem do documento (só texto, em janelas): devolve (índices suspeitos, termos do glossário)."""
    t0 = time.perf_counter()
    n, win = len(texts_a), max(1, int(getattr(config, "quality_triage_window", 60) or 60))
    suspects, terms, calls = set(), [], 0
    for start in range(0, n, win):
        job.check_cancel()
        items = [{"n": i + 1, "B": texts_b[i], "A": texts_a[i]} for i in range(start, min(n, start + win)) if texts_a[i] or texts_b[i]]
        if not items:
            continue
        try:
            r = client.triage(reviewer, items, getattr(job, "context", ""), 20)
        except OllamaError as e:
            job.warnings.append(f"Triagem do documento falhou ({e}); seguindo sem ela.")
            continue
        calls += 1
        valid = {it["n"] for it in items}
        suspects |= {int(x) - 1 for x in r.get("suspects", []) if isinstance(x, int) and x in valid}
        terms += [t for t in r.get("glossary", []) if isinstance(t, str)]
    report.triage = {"calls": calls, "suspects": len(suspects), "terms": len(terms), "seconds": round(time.perf_counter() - t0, 1)}
    return suspects, terms


def _after_generation_v2(job, client, config, chunks, upd, gens, report, session, language, ctx, loops, wd, failures, last_error, rev_end):
    """v2 (docs/plano-pipeline-v2.md): B é a âncora; silêncio é informação; o revisor decide por ponto de divergência."""
    n = len(chunks)
    texts_a, texts_b = gens[0], gens[1]
    min_conf = getattr(config, "quality_min_conf", MIN_CONFIDENCE)
    threshold = getattr(config, "quality_threshold", THRESHOLD)
    signals = judge_v2.file_signals(texts_a, texts_b, language)
    trusted = judge_v2.a_trusted(signals)
    report.judge, report.signals = "v2", {**signals, "a_trusted": trusted}
    if not trusted:
        job.warnings.append("O gerador A parece pouco confiável neste arquivo (escreve mais que o WhisperX ou preenche os silêncios dele): "
                            "ele não altera o texto do WhisperX; o revisor só corrige palavras ouvindo o áudio.")
    frags = []
    for i, ch in enumerate(chunks):
        a, b = texts_a[i], texts_b[i]
        frags.append(Fragment(i, ch.start, ch.end, a=a, b=b, final=b, source="B" if b else ("silêncio" if a else "B")))
    flagged = gate_multi([texts_a, texts_b], language, threshold)
    for i in flagged:
        frags[i].flagged = True
    report.flagged = len(flagged)
    reviewer = _resolve_reviewer(job, client, config, report)
    all_reviews: list[dict] = []
    glossary = specialists.build_glossary(ctx, [])
    review_set = set(flagged)
    if getattr(config, "quality_triage", False):
        suspects, terms = _triage(job, client, config, reviewer, texts_a, texts_b, report)
        review_set |= {i for i in suspects if frags[i].a or frags[i].b}
        glossary = specialists.build_glossary(ctx, [{"terms": terms}])
    order = sorted(review_set)

    # ---------------------------------------------------------------- passada 1: revisão por ponto de divergência
    t0 = time.perf_counter()
    lo, hi = (rev_end - 15, rev_end) if loops else (75, 95)
    for k, i in enumerate(order, 1):
        job.check_cancel()
        upd(lo + (hi - lo) * (k - 1) / max(len(order), 1), f"Revisando fragmento {i + 1} ({k} de {len(order)})")
        f = frags[i]
        f.review = _review_v2(job, client, config, reviewer, frags, chunks, i, f.a, f.b, glossary, loops > 0, trusted)
        report.reviewed += 1
        if f.review:
            all_reviews.append(f.review)
        f.final, f.source = judge_v2.decide(f.a, f.b, f.review, trusted, glossary, min_conf)
        _count_v2(report, f.source)
    report.seconds["revisao"] = round(time.perf_counter() - t0, 1)
    for f in frags:
        f.passes.append({"pass": 1, "a": f.a, "b": f.b, "c": "", "flagged": f.flagged, "review": f.review, "final": f.final, "source": f.source})
    report.passes.append({"pass": 1, "targets": n, "flagged": len(flagged), "reviewed": len(order),
                          "changed": sum(1 for f in frags if f.final != f.b), "seconds": report.seconds["revisao"]})
    report.texts["pass1"] = [f.final for f in frags]
    report.texts["a1"], report.texts["b1"] = list(texts_a), list(texts_b)

    # ---------------------------------------------------------------- refinos: só onde o revisor mexeu no B, com geração INDEPENDENTE
    # (o gerador recebe só o glossário/termos, nunca o texto candidato: assim os erros de A e B continuam independentes)
    targets = [i for i in order if frags[i].final != frags[i].b]
    for it in range(1, loops + 1):
        if not targets:
            break
        job.check_cancel()
        last = it == loops
        glossary = specialists.build_glossary(ctx, [{"terms": [t for r in all_reviews for t in (r.get("terms") or []) if isinstance(r, dict)]}] + [{"terms": glossary}])
        lo = rev_end + (92 - rev_end) * (it - 1) / loops
        span = (92 - rev_end) / loops
        t0 = time.perf_counter()
        note = build_feedback_note("", "", glossary[:12])
        level = str(getattr(config, "quality_feedback", "terms") or "terms")
        before = {i: frags[i].final for i in targets}
        nxt, reviewed = [], 0
        for k, i in enumerate(targets, 1):
            job.check_cancel()
            upd(lo + span * (k - 1) / len(targets), f"Passada {it + 1}: fragmento {i + 1} ({k} de {len(targets)})")
            f = frags[i]
            try:
                wav = audio.to_wav_bytes(chunks[i].samples)
                if level == "terms":  # só termos e nomes: não revela nenhuma escolha
                    a2 = srt.clean_text(client.transcribe(job.model, wav, **({"context": ctx} if ctx else {}), feedback=note))
                else:  # pontos em disputa (sem veredito) e, em "critique", a crítica do revisor; com raciocínio em etapas
                    prev = f.passes[0]
                    crit = str((f.review or {}).get("critique") or "") if level == "critique" else ""
                    a2 = srt.clean_text(client.transcribe_reason(job.model, wav, ctx, judge_v2.word_diffs(prev["a"], prev["b"]), crit, glossary[:12]))
            except OllamaError as e:
                job.warnings.append(f"Passada {it + 1}: gerador A falhou no fragmento {i + 1} ({e}).")
                a2 = f.a
            b2 = quality.collapse_repeats(srt.clean_text(session.transcribe(chunks[i].samples, ", ".join(glossary[:8]) + "." if glossary else ""))) \
                if session else f.b
            agree = bool(a2 and b2) and not judge_v2.word_diffs(a2, b2)  # as duas gerações independentes concordam: nada a revisar
            review = None if agree else _review_v2(job, client, config, reviewer, frags, chunks, i, a2, b2, glossary, not last, trusted)
            reviewed += 0 if agree else 1
            if review:
                all_reviews.append(review)
            new, src = judge_v2.decide(a2, b2, review, trusted, glossary, min_conf)
            if review is None and not agree:  # a revisão falhou: mantém o que já estava decidido
                new, src = f.final, f.source
            f.review, f.final, f.source = review or f.review, new, src
            f.passes.append({"pass": it + 1, "a": a2, "b": b2, "c": "", "flagged": True, "review": review, "final": new, "source": src})
            if new != b2:
                nxt.append(i)
        changed = sum(1 for i in targets if frags[i].final != before[i])
        secs = round(time.perf_counter() - t0, 1)
        report.passes.append({"pass": it + 1, "targets": len(targets), "flagged": len(nxt), "reviewed": reviewed, "changed": changed, "seconds": secs})
        report.seconds[f"refino_{it}"] = secs
        report.loops_done = it
        report.texts[f"pass{it + 1}"] = [f.final for f in frags]
        targets = nxt
        if changed == 0:
            break
    glossary = specialists.build_glossary(ctx, [{"terms": glossary}] + all_reviews)
    return _finish(job, client, config, frags, report, reviewer, glossary, wd, failures, last_error, upd)


MAX_SELF_REWRITE = 0.35  # o gerador não pode reescrever o fragmento além disto (WER contra a tentativa anterior): senão vale a anterior


def _review_self(job, client, config, reviewer, frags, chunks, i, text, glossary, vote_info: dict | None = None) -> dict | None:
    before, after = _neighbours(frags, i, int(getattr(config, "quality_neighbours", 2) or 0))
    ctx = ", ".join(t for t in [getattr(job, "context", "")] + list(glossary[:30]) if t)
    kw = {}
    if vote_info and vote_info.get("doubts"):
        kw["doubts"] = vote_info["doubts"]
    if vote_info and vote_info.get("unstable"):
        kw["unstable"] = True
    try:
        return client.review_self(reviewer, audio.to_wav_bytes(chunks[i].samples), text, chunks[i].end - chunks[i].start,
                                  context=ctx, before=before, after=after, **kw)
    except OllamaError as e:
        _review_failed(job, i + 1, e, "mantido o texto atual")
        return None


def _apply_self_review(text: str, review: dict | None, min_conf: float, vote_info: dict | None = None) -> tuple[str, str]:
    """(texto, origem) depois de aplicar a revisão de UMA transcrição: sem fala = silêncio; senão as escolhas nos pontos de dúvida da
    votação (montadas pelo código) e as correções de palavra, com trava de semelhança."""
    if not review:
        return text, "A"
    try:
        conf = float(review.get("confidence", 1.0) or 0)
    except (TypeError, ValueError):
        conf = 0.0
    if conf < min_conf:
        return text, "A"
    if review.get("speech") is False and text:
        return "", "silêncio"
    src = "A"
    doubts, picks = (vote_info or {}).get("doubts"), review.get("picks")
    if doubts and isinstance(picks, list) and len(picks) == len(doubts):
        chosen = vote.apply_choices(text, vote_info["alt"], picks)
        if chosen != text:
            text, src = chosen, "A+voto"
    new, n = judge_v2.apply_fixes(text, review.get("fixes"), [t for t in (review.get("terms") or []) if isinstance(t, str)])
    return new, src + "+correção" if n else src


def _vote(job, client, config, chunks, base: list[str], ctx: str, report: QualityReport, upd) -> tuple[list[str], dict]:
    """Votação do modo 1: 2 hipóteses extras do gerador por fragmento com texto; devolve (consenso por fragmento, {i: dúvidas}).
    Só chama o gerador (o revisor ouve os pontos de dúvida depois)."""
    t0 = time.perf_counter()
    out, info = list(base), {}
    idx = [i for i in range(len(chunks)) if base[i]]
    stats = {"fragments": len(idx), "applied": 0, "with_doubts": 0, "doubts": 0, "unstable": 0, "silenced": 0, "failed": 0}
    for k, i in enumerate(idx, 1):
        job.check_cancel()
        upd(40.0 + 5.0 * (k - 1) / max(len(idx), 1), f"Votação: fragmento {i + 1} ({k} de {len(idx)})")
        hyps = [base[i]]
        for kind, samples in (("literal", chunks[i].samples), ("careful", audio.normalize_peak(chunks[i].samples))):
            try:
                hyps.append(srt.clean_text(client.transcribe_variant(job.model, audio.to_wav_bytes(samples), ctx, kind)))
            except OllamaError as e:
                stats["failed"] += 1
                job.warnings.append(f"Hipótese '{kind}' do fragmento {i + 1} falhou ({e}).")
        if len(hyps) < 2:
            continue
        c = vote.consensus(hyps)
        out[i] = c["text"]
        stats["applied"] += c["applied"]
        unstable = c["stability"] < vote.UNSTABLE_BELOW
        stats["unstable"] += unstable
        if not c["text"]:
            stats["silenced"] += 1
            continue
        dl = vote.doubts_of(c["text"], c["alt"]) if c["alt"] != c["text"] else []
        if len(dl) > vote.MAX_DOUBTS:
            dl, unstable = [], True
        if dl or unstable:
            info[i] = {"alt": c["alt"], "doubts": dl, "unstable": unstable}
            stats["with_doubts"] += bool(dl)
            stats["doubts"] += len(dl)
    stats["seconds"] = round(time.perf_counter() - t0, 1)
    report.vote = stats
    report.seconds["votacao"] = stats["seconds"]
    return out, info


def _after_generation_self(job, client, config, chunks, upd, texts_a, report, language, ctx, loops, wd, failures, last_error):
    """Modo de qualidade 1: autorrefinamento gerador <-> revisor, SEM WhisperX (docs/experimento-loop-refinamento.md).

    Passada 1: o gerador transcreve tudo. Cada rodada: o revisor (com áudio) julga cada transcrição (raciocínio em etapas, veredito ok/fix,
    correções de palavras, crítica); quem não está ok volta ao gerador (tentativa anterior + crítica + termos), com trava contra reescrita
    excessiva; sem fala = silêncio. As correções de palavra do revisor valem em cada rodada (o que sai de cada uma é a `passN`)."""
    n = len(chunks)
    lang = _lang(language)
    min_conf = getattr(config, "quality_min_conf", MIN_CONFIDENCE)
    reviewer = _resolve_reviewer(job, client, config, report)
    report.judge = "self"
    frags = [Fragment(i, ch.start, ch.end, a=texts_a[i], final=texts_a[i], source="A") for i, ch in enumerate(chunks)]
    cur = list(texts_a)  # tentativa atual de cada fragmento (a que o gerador escreveu)
    glossary: list[str] = specialists.build_glossary(ctx, [])
    all_reviews: list[dict] = []
    report.texts["a1"] = list(texts_a)
    vote_info: dict = {}
    if getattr(config, "quality_vote", False):  # 3 hipóteses do gerador -> consenso + dúvidas para o revisor
        cur, vote_info = _vote(job, client, config, chunks, cur, ctx, report, upd)
        report.texts["consenso"] = list(cur)
        for i in range(n):
            if texts_a[i] and not cur[i]:  # a maioria das hipóteses ouviu silêncio: o fragmento não vai ao revisor
                frags[i].final, frags[i].source = "", "silêncio"
    active = [i for i in range(n) if cur[i]]
    report.flagged = len(active)
    for rnd in range(loops + 1):
        last = rnd == loops
        t0 = time.perf_counter()
        lo, hi = (45.0 + 47.0 * rnd / (loops + 1), 45.0 + 47.0 * (rnd + 1) / (loops + 1))
        need_fix = []
        for k, i in enumerate(active, 1):
            job.check_cancel()
            upd(lo + (hi - lo) * 0.5 * (k - 1) / max(len(active), 1), f"Rodada {rnd + 1}: revisando fragmento {i + 1} ({k} de {len(active)})")
            f = frags[i]
            vi = vote_info.get(i) if rnd == 0 else None  # as dúvidas da votação só valem para o texto da votação (1ª rodada)
            r = _review_self(job, client, config, reviewer, frags, chunks, i, cur[i], glossary, vi)
            report.reviewed += 1
            f.review = r or f.review
            if r:
                all_reviews.append(r)
                for t in r.get("terms") or []:
                    if isinstance(t, str) and t.strip() and t.strip() not in glossary:
                        glossary.append(t.strip())
            text, src = _apply_self_review(cur[i], r, min_conf, vi)
            if "voto" in src and text:
                cur[i] = text  # o refino parte do texto com as escolhas já aplicadas
            f.final, f.source = text, src
            f.passes.append({"pass": rnd + 1, "a": cur[i], "b": "", "c": "", "flagged": True, "review": r, "final": text, "source": src})
            if r and str(r.get("verdict")).lower() == "fix" and text:
                need_fix.append(i)
        report.texts[f"pass{rnd + 1}"] = [f.final for f in frags]
        changed = sum(1 for f in frags if f.final != texts_a[f.index])
        report.passes.append({"pass": rnd + 1, "targets": len(active), "flagged": len(need_fix), "reviewed": len(active), "changed": changed,
                              "seconds": round(time.perf_counter() - t0, 1)})
        report.loops_done = rnd
        if last or not need_fix:
            break
        # refino: o gerador reescuta e reescreve só o que o revisor apontou
        for k, i in enumerate(need_fix, 1):
            job.check_cancel()
            upd(lo + (hi - lo) * (0.5 + 0.5 * (k - 1) / len(need_fix)), f"Rodada {rnd + 1}: refinando fragmento {i + 1} ({k} de {len(need_fix)})")
            r = frags[i].review or {}
            try:
                new = srt.clean_text(client.transcribe_self(job.model, audio.to_wav_bytes(chunks[i].samples), ctx, cur[i],
                                                            str(r.get("critique") or ""), [t for t in (r.get("terms") or []) if isinstance(t, str)]))
            except OllamaError as e:
                job.warnings.append(f"Refino do fragmento {i + 1} falhou ({e}); mantida a tentativa anterior.")
                continue
            if new and metrics.wer(cur[i], new, lang) <= MAX_SELF_REWRITE:
                cur[i] = new
        active = need_fix
    report.seconds["autorrefino"] = round(sum(p["seconds"] for p in report.passes), 1)
    for f in frags:
        _count_v2(report, f.source)
    glossary = specialists.build_glossary(ctx, [{"terms": glossary}] + all_reviews)
    return _finish(job, client, config, frags, report, reviewer, glossary, wd, failures, last_error, upd)


def _count(report: QualityReport, src: str) -> None:
    if src == "A":
        report.from_a += 1
    elif src == "B":
        report.from_b += 1
    elif src == "C":
        report.from_c += 1
    elif src == "merge":
        report.merged += 1
    else:
        report.kept_a += 1


def review_translation_step(job, client, config, source: list[str], translated: list[str], upd=None) -> list[str]:
    """Especialista de tradução, depois da tradução e antes da legenda: só nas falas que de fato foram traduzidas."""
    if not getattr(config, "quality_specialists", False):
        return translated
    idx = [i for i, (s, t) in enumerate(zip(source, translated)) if s and t and s != t]
    if not idx:
        return translated
    model = getattr(job, "report", {}) and job.report.get("reviewer") or job.model
    t0 = time.perf_counter()
    if upd:
        upd(96, "Especialista: revisando a tradução")
    glossary = ", ".join((job.report or {}).get("specialists", {}).get("glossary", [])[:30]) if getattr(job, "report", None) else ""
    new, stats = specialists.review_translation(client, model, [(source[i], translated[i]) for i in idx], job.language, glossary, check=job.check_cancel)
    out = list(translated)
    for i, t in zip(idx, new):
        out[i] = t
    if getattr(job, "report", None) is not None:
        job.report.setdefault("specialists", {})["translation"] = {**stats, "model": model, "seconds": round(time.perf_counter() - t0, 1)}
    return out
