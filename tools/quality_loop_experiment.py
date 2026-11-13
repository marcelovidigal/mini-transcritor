"""Experimento do laço de refinamento da opção Qualidade: geradores A × modos (revisor e4b / 12b), B = WhisperX, 3 passadas.

    python tools/quality_loop_experiment.py filme.json --cache cache/ --out resultado/ --generators gemma4:e2b gemma4:e4b gemma4:12b ^
        --modes quality-e4b quality-12b --translate

Roda o motor de verdade (`quality_engine.run` com a config dos modos do app) sobre o cache de chamadas: cada chamada ao Ollama/WhisperX
(transcrição, revisão, pontuação, tradução) é guardada por (método, modelo, entrada) junto com o tempo gasto, então reexecutar é
barato, a 1ª geração (A e B) é compartilhada entre os modos e o tempo de cada combinação é a soma dos tempos registrados
("tempo equivalente"). RODE SÓ COM A GPU LIVRE; parallel=1 (somas de tempo corretas).

Por combinação e por clipe mede o WER (ou, no filme, o WER do inglês falado) de: A sozinho, B sozinho, passada 1, 2, 3, antes dos
especialistas e final; com --translate também chrF/WER do pt-BR contra as referências traduzidas e grava a legenda (.srt) para o juiz.
Retoma de onde parou (resultados em --out/resultados.json)."""
import argparse
import hashlib
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    _s.reconfigure(encoding="utf-8", errors="replace")

from transcritor import audio, engines, metrics, pipeline, quality, srt, workdir  # noqa: E402
from transcritor import quality_engine as qe  # noqa: E402
from transcritor.config import Config  # noqa: E402
from transcritor.jobs import Job  # noqa: E402
from transcritor.ollama_client import OllamaClient  # noqa: E402


def sha(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else json.dumps(p, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class Cache:
    """Chamadas guardadas em disco: {chave: {"v": valor, "t": segundos}}; thread-safe; grava a cada 25 novas entradas."""

    def __init__(self, path):
        self.path, self.lock, self.new = path, threading.RLock(), 0
        self.data = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
        self.spent: dict[str, float] = {}

    def reset(self):
        self.spent = {}

    def get(self, key, label, fn):
        with self.lock:
            hit = self.data.get(key)
        if hit is None:
            t0 = time.perf_counter()
            val = fn()
            hit = {"v": val, "t": round(time.perf_counter() - t0, 3)}
            with self.lock:
                self.data[key] = hit
                self.new += 1
                if self.new % 25 == 0:
                    self.save()
        with self.lock:
            self.spent[label] = self.spent.get(label, 0.0) + hit["t"]
        return hit["v"]

    def save(self):
        with self.lock:
            tmp = self.path + ".tmp"
            json.dump(self.data, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
            os.replace(tmp, self.path)


class CachingClient:
    """Mesma interface do OllamaClient, com cache. O rótulo do tempo gasto identifica a fase (transcribe, review, polish…)."""

    def __init__(self, inner, cache):
        self.inner, self.cache = inner, cache
        self.keep_alive = inner.keep_alive

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def transcribe(self, model, wav, context="", feedback=""):
        label = "A_refino" if feedback else "A_geracao"
        return self.cache.get(sha("transcribe", model, wav, context, feedback), label, lambda: self.inner.transcribe(model, wav, context, feedback))

    def review(self, model, wav, a, b, context="", c="", **kw):
        return self.cache.get(sha("review", model, wav, a, b, context, c, kw), "revisao", lambda: self.inner.review(model, wav, a, b, context, c, **kw))

    def review_v2(self, model, wav, a, b, diffs, seconds, context="", before="", after="", notes=False, trusted=False):
        # trusted só entra na chave quando verdadeiro: as revisões com o prompt "âncora" gravadas antes continuam valendo
        key = ("review_v2", model, wav, a, b, diffs, context, before, after, notes) + (("neutral",) if trusted else ())
        return self.cache.get(sha(*key), "revisao",
                              lambda: self.inner.review_v2(model, wav, a, b, diffs, seconds, context, before, after, notes, trusted))

    def transcribe_reason(self, model, wav, context="", points=None, critique="", terms=None):
        return self.cache.get(sha("reason", model, wav, context, points, critique, terms), "A_refino",
                              lambda: self.inner.transcribe_reason(model, wav, context, points, critique, terms))

    def review_self(self, model, wav, text, seconds, context="", before="", after="", doubts=None, unstable=False):
        # doubts/unstable só entram na chave quando existem: as revisões gravadas antes da votação continuam valendo
        key = ("review_self", model, wav, text, context, before, after) + ((doubts, unstable) if (doubts or unstable) else ())
        return self.cache.get(sha(*key), "revisao",
                              lambda: self.inner.review_self(model, wav, text, seconds, context, before, after, doubts, unstable))

    def transcribe_variant(self, model, wav, context="", kind="literal"):
        return self.cache.get(sha("variant", model, wav, context, kind), "A_votacao", lambda: self.inner.transcribe_variant(model, wav, context, kind))

    def transcribe_self(self, model, wav, context="", previous="", critique="", terms=None):
        return self.cache.get(sha("transcribe_self", model, wav, context, previous, critique, terms), "A_refino",
                              lambda: self.inner.transcribe_self(model, wav, context, previous, critique, terms))

    def triage(self, model, items, context="", limit=20):
        return self.cache.get(sha("triage", model, items, context, limit), "triagem", lambda: self.inner.triage(model, items, context, limit))

    def polish(self, model, texts, language="", glossary=""):
        return self.cache.get(sha("polish", model, texts, language, glossary), "pontuacao", lambda: self.inner.polish(model, texts, language, glossary))

    def review_translation(self, model, pairs, language, glossary=""):
        return self.cache.get(sha("rtr", model, pairs, language, glossary), "rev_traducao", lambda: self.inner.review_translation(model, pairs, language, glossary))

    def translate_batch(self, model, texts, language, context=""):
        return self.cache.get(sha("tb", model, texts, language, context), "traducao", lambda: self.inner.translate_batch(model, texts, language, context))

    def translate(self, model, text, language, context=""):
        return self.cache.get(sha("t1", model, text, language, context), "traducao", lambda: self.inner.translate(model, text, language, context))


class CachedSession:
    """Sessão do WhisperX com cache: só carrega o modelo de verdade se alguma chamada não estiver no cache."""

    def __init__(self, factory, cache, lang_key):
        self.factory, self.cache, self.real, self.lang_key = factory, cache, None, lang_key

    def transcribe(self, samples, prompt=""):
        def real():
            if self.real is None:
                self.real = self.factory()
            return self.real.transcribe(samples, prompt)

        return self.cache.get(sha("wx", samples.tobytes(), prompt, self.lang_key), "B_refino" if prompt else "B_geracao", real)

    def close(self):
        if self.real is not None:
            self.real.close()
            self.real = None


def install_cached_whisperx(cache, model_name):
    """Troca o `session` do motor WhisperX por uma versão com cache (o motor real só é usado nas falhas de cache)."""
    eng = engines.get("whisperx")
    orig = eng.session

    def session(model, *, device="auto", context="", job=None):
        return CachedSession(lambda: orig(model, device=device, context=context, job=job), cache, [model, context])

    eng.session = session


def free_ollama(host):
    try:
        OllamaClient(host).unload_all()
    except Exception:
        pass


def read(path):
    return open(path, encoding="utf-8").read()


def wer_of(ref, texts, lang):
    return round(metrics.wer(ref, " ".join(t for t in texts if t), lang), 4)


def to_srt(chunks, texts):
    cues = []
    for ch, t in zip(chunks, texts):
        if t:
            cues.extend(srt.cues_for_chunk(ch.start, ch.end, t))
    return srt.apply_style(cues)


def prepare(item, workdir_tmp):
    wav = os.path.join(workdir_tmp, "audio.wav")
    audio.extract_wav(item["audio"], wav)
    samples = audio.read_wav(wav)
    return samples, audio.split_by_speech(samples, 12.0, 28.0)


def run_combo(item, stem, a_model, mode, samples, chunks, client, cache, args, outdir):
    base = Config(parallel=1, translate_batch=8, translate_model=args.translate_model, work_dir=os.path.join(outdir, "work"), quality_b_model=args.b_model)
    if mode.startswith("self-"):  # modo de qualidade 1: autorrefinamento gerador <-> revisor, sem WhisperX; "self-<revisor>"
        base.quality_self, base.quality_reviewer, base.quality_specialists = True, mode[5:], args.specialists
        base.quality_gate, base.quality_vote = args.gate, args.vote
        cfg = base
    else:
        cfg = qe.apply_mode(mode, base)
        cfg.quality_b_model = args.b_model
    cfg.quality_judge, cfg.quality_triage, cfg.quality_feedback = args.judge, args.triage, args.feedback
    cfg.quality_loops = args.loops  # os modos do app usam 2; --loops 1 = 2 passadas no total
    job = Job(id=f"exp-{stem}-{a_model.replace(':', '_')}-{mode.replace(':', '_')}", path=item["audio"], language=item["language"], model=a_model, engine=mode,
              context=item.get("context", ""), source_name=stem)
    cache.reset()
    t0 = time.perf_counter()
    texts, failures, _err, rep = qe.run(job, client, cfg, samples, chunks, lambda p, s=None: None, pipeline._transcribe_all)
    wall = time.perf_counter() - t0
    workdir.remove(job.work_dir)
    spent = dict(cache.spent)
    lang_src = item.get("language_src") or item["language"]
    ref_src = read(item.get("reference_src") or item["reference"])
    row = {"clip": stem, "a": a_model, "mode": mode, "reviewer": rep.reviewer, "fragments": len(chunks), "failures": failures,
           "gate": args.gate, "vote": args.vote, "gate_info": rep.gate, "vote_info": rep.vote, "loops": args.loops, "loops_done": rep.loops_done, "passes": rep.passes, "specialists": rep.specialists, "sources": rep.sources,
           "counts": {"flagged": rep.flagged, "from_a": rep.from_a, "from_b": rep.from_b, "merged": rep.merged, "kept": rep.kept_a},
           "seconds_equiv": {k: round(v, 1) for k, v in spent.items()}, "seconds_equiv_total": round(sum(spent.values()), 1),
           "wall_s": round(wall, 1), "warnings": job.warnings[-5:]}
    t = rep.texts
    row["wer_src"] = {"A": wer_of(ref_src, t["a1"], lang_src), "B": (wer_of(ref_src, t["b1"], lang_src) if "b1" in t else None),
                      "pass1": wer_of(ref_src, t["pass1"], lang_src)}
    for k in sorted(k for k in t if k.startswith("pass") and k != "pass1"):
        row["wer_src"][k] = wer_of(ref_src, t[k], lang_src)
    if "consenso" in t:
        row["wer_src"]["consenso"] = wer_of(ref_src, t["consenso"], lang_src)
    if "pre_specialists" in t:
        row["wer_src"]["pre_specialists"] = wer_of(ref_src, t["pre_specialists"], lang_src)
    row["wer_src"]["final"] = wer_of(ref_src, t["final"], lang_src)
    row["judge"], row["feedback"], row["triage"], row["signals"], row["silenced"] = args.judge, args.feedback, rep.triage, rep.signals, rep.silenced
    row["breakdown"] = {k: metrics.error_breakdown(ref_src, " ".join(x for x in t[key] if x), lang_src)
                        for k, key in (("A", "a1"), ("B", "b1"), ("consenso", "consenso"), ("pass1", "pass1"), ("final", "final")) if key in t}
    json.dump(t, open(os.path.join(outdir, "textos", f"{stem}.{a_model.replace(':', '_')}.{mode.replace(':', '_')}.json"), "w", encoding="utf-8"), ensure_ascii=False)
    if args.translate and item.get("reference"):
        row["translation"] = translate_phase(item, stem, a_model, mode, job, cfg, rep, t, chunks, client, cache, args, outdir)
    return row


def translate_phase(item, stem, a_model, mode, job, cfg, rep, t, chunks, client, cache, args, outdir):
    """Traduz a passada 1 (linha de base), o final sem o especialista de tradução e o final com ele; mede pt-BR e grava os SRTs."""
    refs = {"e4b": read(item["reference"])}
    if item.get("reference_12b"):
        refs["12b"] = read(item["reference_12b"])
    out, tag = {}, f"{stem}.{a_model.replace(':', '_')}.{mode.replace(':', '_')}"
    job.report = rep.as_dict()
    sets = {"pass1": t["pass1"], "final": t["final"]}
    for name, src in sets.items():
        cache.reset()
        tr = pipeline._translate_all(job, client, list(src), cfg, lambda p, s=None: None, model=args.translate_model)
        variants = {name: tr}
        if name == "final":
            variants["final+trad"] = qe.review_translation_step(job, client, cfg, list(src), tr, None)
        for vname, texts in variants.items():
            cues = to_srt(chunks, texts)
            open(os.path.join(outdir, "saidas", f"{tag}.{vname}.srt"), "w", encoding="utf-8", newline="\n").write(srt.to_srt(cues))
            hyp = " ".join(c.text.replace("\n", " ") for c in cues)
            m = {f"chrf_{k}": round(metrics.chrf(ref, hyp), 4) for k, ref in refs.items()}
            m.update({f"wer_{k}": round(metrics.wer(ref, hyp, "pt-BR"), 4) for k, ref in refs.items()})
            m["cps_over"] = metrics.rule_stats(cues)["over_cps"]
            out[vname] = m
        out["_seconds_" + name] = {k: round(v, 1) for k, v in cache.spent.items()}
    if (job.report or {}).get("specialists", {}).get("translation"):
        out["translation_specialist"] = job.report["specialists"]["translation"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--generators", nargs="+", default=["gemma4:e2b", "gemma4:e4b", "gemma4:12b"])
    ap.add_argument("--modes", nargs="+", default=["quality-e4b", "quality-12b"])
    ap.add_argument("--loops", type=int, default=2, help="refinamentos depois da 1ª passada (2 = 3 passadas, como os modos do app)")
    ap.add_argument("--judge", choices=["v1", "v2"], default="v1", help="v2 = B âncora, silêncio, decisão por ponto (docs/plano-pipeline-v2.md)")
    ap.add_argument("--feedback", choices=["terms", "points", "critique"], default="terms", help="v2, refino: o que o gerador A recebe (termos | + pontos em disputa | + crítica)")
    ap.add_argument("--triage", action="store_true", help="v2: triagem do documento inteiro (só texto) antes da revisão")
    ap.add_argument("--pairs", nargs="+", default=[], help="modo 1 (autorrefinamento, sem WhisperX): pares gerador/revisor, ex.: gemma4:e2b/gemma4:e4b")
    ap.add_argument("--gate", action="store_true", help="modo 1: porteiro de fala (Silero) antes do gerador")
    ap.add_argument("--vote", action="store_true", help="modo 1: votação entre 3 hipóteses do gerador + dúvidas para o revisor")
    ap.add_argument("--specialists", action="store_true", help="modo 1: liga os especialistas (termos, pontuação, tradução)")
    ap.add_argument("--b-model", default="large-v3-turbo")
    ap.add_argument("--translate", action="store_true", help="também traduz (filme: en → pt-BR) e mede o pt-BR")
    ap.add_argument("--translate-model", default="gemma4:e4b")
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    for sub in ("textos", "saidas", "work"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    res_path = os.path.join(args.out, "resultados.json")
    results = json.load(open(res_path, encoding="utf-8")) if os.path.exists(res_path) else []
    done = {(r["clip"], r["a"], r["mode"]) for r in results}
    host = Config().ollama_host
    for mpath in args.manifests:
        for item in json.loads(read(mpath)):
            stem = os.path.splitext(os.path.basename(item["audio"]))[0]
            cache = Cache(os.path.join(args.cache, stem + ".json"))
            install_cached_whisperx(cache, args.b_model)
            client = CachingClient(OllamaClient(host, 300), cache)
            import tempfile

            with tempfile.TemporaryDirectory(prefix="mt_loop_") as tmp:
                samples, chunks = prepare(item, tmp)
            print(f"== {stem}: {len(chunks)} fragmentos", flush=True)
            combos = [(p.split("/")[0], "self-" + p.split("/")[1]) for p in args.pairs] if args.pairs else                 [(a, m) for a in args.generators for m in args.modes]
            for a_model, mode in combos:
                if True:
                    if (stem, a_model, mode) in done:
                        continue
                    free_ollama(host)
                    print(f"→ A={a_model} modo={mode} ({time.strftime('%H:%M:%S')})", flush=True)
                    try:
                        row = run_combo(item, stem, a_model, mode, samples, chunks, client, cache, args, args.out)
                    except Exception as e:  # uma falha não derruba o resto
                        import traceback

                        traceback.print_exc()
                        row = {"clip": stem, "a": a_model, "mode": mode, "error": str(e)}
                    cache.save()
                    results.append(row)
                    json.dump(results, open(res_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                    if "error" not in row:
                        w = row["wer_src"]
                        print(f"   WER A {w['A']:.4f} B {w['B'] or 0:.4f} | p1 {w['pass1']:.4f} p2 {w.get('pass2', 0):.4f} p3 {w.get('pass3', 0):.4f} "
                              f"| final {w['final']:.4f} | {row['seconds_equiv_total']}s equiv", flush=True)
            cache.save()


if __name__ == "__main__":
    main()
