"""Matriz de avaliação da opção "Qualidade": todos os PARES de geradores × revisores, com cache — RODE SÓ COM A GPU LIVRE.

    python tools/quality_matrix.py real.json holdout.json --cache cache/ --out resultado/ ^
        --reviewers gemma4:e4b nemotron-3.5-bolt:e4b-q4_K_M gemma4:12b

Cada manifesto é uma lista JSON de {"audio", "language", "reference"} (ver tools/evaluate.py). Regras (confirmadas pelo usuário):
  * o pipeline tem SÓ 2 geradores (G1, G2); o Whisper só gera (nunca revisa); o revisor é um modelo de áudio do Ollama;
  * em caso de dúvida/sem decisão do revisor vale a geração G2 (a mais forte do par: Whisper, ou e4b sobre e2b);
  * cada gerador roda UMA vez por clipe (cache em --cache); cada combinação só refaz portão, revisão e juiz.
Saída: matriz.md (tabelas) e matriz.json. Também faz a comparação direta de dois modelos Ollama (--compare), para saber
se um deles é redundante.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    _s.reconfigure(encoding="utf-8", errors="replace")

from transcritor import audio, engines, metrics, quality, srt  # noqa: E402
from transcritor import quality_engine as qe  # noqa: E402
from transcritor.config import Config  # noqa: E402
from transcritor.jobs import Job  # noqa: E402
from transcritor.ollama_client import OllamaClient, OllamaError  # noqa: E402

DEFAULT_GENS = ["ollama:gemma4:e2b", "ollama:gemma4:e4b", "whisper:medium", "whisper:large-v3", "whisper:large-v3-turbo"]
DEFAULT_PAIRS = [("ollama:gemma4:e2b", "ollama:gemma4:e4b")] + [
    (a, w) for a in ("ollama:gemma4:e2b", "ollama:gemma4:e4b") for w in ("whisper:medium", "whisper:large-v3", "whisper:large-v3-turbo")]


def short(g: str) -> str:
    return g.split(":", 1)[1].replace("gemma4:", "").replace("nemotron-3.5-bolt:e4b-q4_K_M", "nemotron").replace("large-v3-turbo", "turbo").replace("large-v3", "large") \
        if g.startswith(("ollama:", "whisper:")) else g


def label(g: str) -> str:
    kind, name = g.split(":", 1)
    return ("W-" if kind == "whisper" else "") + short(g)


def read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def prepare(item, cfg):
    """Fragmentos de fala (mesmo VAD do app) e referência do clipe."""
    with tempfile.TemporaryDirectory(prefix="mt_matrix_") as tmp:
        wav = os.path.join(tmp, "a.wav")
        audio.extract_wav(item["audio"], wav)
        samples = audio.read_wav(wav)
    chunks = audio.split_by_speech(samples, cfg.chunk_seconds, cfg.max_chunk_seconds)
    return samples, chunks, read_json_text(item["reference"])


def read_json_text(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


class Stub(Job):
    pass


def free_ollama(client):
    """Descarrega os modelos do Ollama da memória da GPU (o Whisper na XPU precisa de VRAM livre; sem isto deu OUT_OF_RESOURCES)."""
    import requests

    try:
        for m in requests.get(client.host + "/api/ps", timeout=10).json().get("models", []):
            requests.post(client.host + "/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=60)
    except Exception:
        pass
    time.sleep(2)


def generate(gen: str, clip: str, item, cfg, client, cache_dir):
    """Textos por fragmento de um gerador num clipe (cache em disco). Devolve (textos, segundos)."""
    path = os.path.join(cache_dir, f"{clip}__{gen.replace(':', '_').replace('/', '_')}.json")
    if os.path.exists(path):
        d = read_json(path)
        return d["texts"], d["seconds"]
    samples, chunks, _ref = prepare(item, cfg)
    kind, model = gen.split(":", 1)
    if kind == "whisper":
        eng = engines.get("openai-whisper")
        free_ollama(client)
        if not eng.is_installed(model):
            eng.ensure_model(model)  # o download não entra no tempo medido
    t0 = time.perf_counter()
    if kind == "ollama":
        texts = []
        for ch in chunks:
            try:
                texts.append(srt.clean_text(client.transcribe(model, audio.to_wav_bytes(ch.samples))))
            except OllamaError as e:
                print(f"    ! {gen} {clip}: {e}", flush=True)
                texts.append("")
    else:
        job = Stub(id="m", path="", language="pt-BR", model=model, engine="openai-whisper")
        segs, _lang = eng.transcribe(model, samples, chunks, job, lambda *a, **k: None)
        texts = [quality.collapse_repeats(t) for t in qe.group_by_fragment(segs, chunks)]
    seconds = round(time.perf_counter() - t0, 1)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"texts": texts, "seconds": seconds, "starts": [c.start for c in chunks], "ends": [c.end for c in chunks]}, f, ensure_ascii=False)
    return texts, seconds


def sign_test_p(wins: int, losses: int) -> float:
    """p bilateral exato do teste dos sinais (empates ignorados)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def final_wer(ref, texts, durations):
    cleaned, _rep = quality.clean_texts(texts, durations)
    return metrics.wer(ref, " ".join(t for t in cleaned if t), "pt"), cleaned


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reviewers", nargs="+", default=["gemma4:e4b"])
    ap.add_argument("--threshold", type=float, default=qe.THRESHOLD)
    ap.add_argument("--gens", nargs="*", default=None, help="geradores extra além dos dos pares (ex.: ollama:nemotron-3.5-bolt:e4b-q4_K_M)")
    ap.add_argument("--compare", nargs=2, default=None, metavar=("MODELO_A", "MODELO_B"), help="comparação direta de dois modelos Ollama")
    ap.add_argument("--no-reviews", action="store_true", help="só gera e compara geradores (sem as combinações com revisor)")
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    cfg = Config()
    client = OllamaClient(cfg.ollama_host, cfg.ollama_timeout)
    clips = []
    for si, mpath in enumerate(args.manifests):
        for item in read_json(mpath):
            clips.append((f"s{si}_" + os.path.splitext(os.path.basename(item["audio"]))[0], item, f"S{si}"))
    gens = list(DEFAULT_GENS) + [g for g in (args.gens or []) if g not in DEFAULT_GENS]
    if args.compare:
        for m in args.compare:
            g = f"ollama:{m}"
            if g not in gens:
                gens.append(g)

    # 1) gerações (um gerador por vez: o modelo fica carregado)
    print(f"== Gerações: {len(gens)} geradores × {len(clips)} clipes", flush=True)
    data = {}  # (clip, gen) -> (textos, segundos)
    meta = {}
    for g in gens:
        for clip, item, _s in clips:
            t0 = time.perf_counter()
            data[(clip, g)] = generate(g, clip, item, cfg, client, args.cache)
            print(f"   {label(g):12} {clip:18} {data[(clip, g)][1]:6.1f}s  ({time.perf_counter() - t0:.0f}s)", flush=True)
    for clip, item, _s in clips:
        d = read_json(os.path.join(args.cache, f"{clip}__{gens[0].replace(':', '_')}.json"))
        ref = read_json_text(item["reference"])
        meta[clip] = {"ref": ref, "durations": [e - s for s, e in zip(d["starts"], d["ends"])], "words": len(metrics.normalize(ref, "pt")),
                      "audio_s": d["ends"][-1] if d["ends"] else 0.0}

    sets = sorted({s for _c, _i, s in clips})
    clips_of = {s: [c for c, _i, ss in clips if ss == s] for s in sets}
    W = {s: sum(meta[c]["words"] for c in clips_of[s]) for s in sets}
    Wtot = sum(W.values())
    AUD = sum(meta[c]["audio_s"] for c in meta)

    def wer_set(texts_by_clip: dict) -> dict:
        out, tot = {}, 0.0
        for s in sets:
            e = 0.0
            for c in clips_of[s]:
                w, _ = final_wer(meta[c]["ref"], texts_by_clip[c], meta[c]["durations"])
                e += w * meta[c]["words"]
            out[s] = e / W[s]
            tot += e
        out["total"] = tot / Wtot
        return out

    results = {"singles": {}, "pairs": []}
    # 2) geradores isolados
    for g in gens:
        wers = wer_set({c: data[(c, g)][0] for c, _i, _s in clips})
        secs = sum(data[(c, g)][1] for c, _i, _s in clips)
        results["singles"][g] = {"wer": wers, "rtf": round(secs / AUD, 3)}

    # 3) comparação direta entre dois modelos Ollama
    if args.compare:
        ga, gb = (f"ollama:{m}" for m in args.compare)
        wins_a = wins_b = ties = ident = nfrag = 0
        between_e = between_w = 0.0
        for c, _i, _s in clips:
            ta, tb = data[(c, ga)][0], data[(c, gb)][0]
            ea = metrics.fragment_errors(meta[c]["ref"], ta, "pt")
            eb = metrics.fragment_errors(meta[c]["ref"], tb, "pt")
            for i in range(len(ta)):
                nfrag += 1
                ident += metrics.normalize(ta[i], "pt") == metrics.normalize(tb[i], "pt")
                d = ea[i]["errors"] - eb[i]["errors"]
                wins_a += d < 0
                wins_b += d > 0
                ties += d == 0
            between_e += metrics.wer(" ".join(ta), " ".join(tb), "pt") * meta[c]["words"]
            between_w += meta[c]["words"]
        results["compare"] = {"a": args.compare[0], "b": args.compare[1], "fragments": nfrag, "identical": ident,
                              "wins_a": wins_a, "wins_b": wins_b, "ties": ties, "p_sign": round(sign_test_p(wins_a, wins_b), 4),
                              "wer_between": round(between_e / between_w, 4)}

    # 4) pares × revisores
    if not args.no_reviews:
        cache_rev = {}
        rev_path = os.path.join(args.cache, "reviews.json")
        if os.path.exists(rev_path):
            cache_rev = read_json(rev_path)

        def review(reviewer, clip, i, ta, tb):
            key = hashlib.sha1(f"{reviewer}|{clip}|{i}|{ta}|{tb}".encode()).hexdigest()
            if key in cache_rev:
                return cache_rev[key]
            wavb = read_wav_fragment(clip, i)
            t0 = time.perf_counter()
            try:
                r = client.review(reviewer, wavb, ta, tb)
            except OllamaError as e:
                r = None
                print(f"    ! revisão {reviewer} {clip}#{i}: {str(e)[:80]}", flush=True)
            cache_rev[key] = {"r": r, "s": round(time.perf_counter() - t0, 2)}
            return cache_rev[key]

        frag_wavs = {}

        def read_wav_fragment(clip, i):
            if clip not in frag_wavs:
                item = next(it for c, it, _s in clips if c == clip)
                samples, chunks, _ = prepare(item, cfg)
                frag_wavs[clip] = [audio.to_wav_bytes(ch.samples) for ch in chunks]
            return frag_wavs[clip][i]

        for reviewer in args.reviewers:
            print(f"== Revisor {reviewer}", flush=True)
            for g1, g2 in DEFAULT_PAIRS:
                finals_by_clip, flagged_n, frag_n, rev_secs = {}, 0, 0, 0.0
                imp = wor = same = 0
                orac_all_e = orac_gate_e = base_e = 0
                for c, _i, _s in clips:
                    ta, tb = data[(c, g1)][0], data[(c, g2)][0]
                    n = len(ta)
                    flagged = set(qe.gate(ta, tb, "pt-BR", args.threshold))
                    finals = [tb[i] if tb[i] else ta[i] for i in range(n)]  # prioridade: G2
                    base = list(finals)
                    for i in sorted(flagged):
                        rv = review(reviewer, c, i, ta[i], tb[i])
                        rev_secs += rv["s"]
                        finals[i], _src = qe.judge(ta[i], tb[i], rv["r"], "pt-BR", "", qe.MIN_CONFIDENCE, "B")
                    finals_by_clip[c] = finals
                    flagged_n += len(flagged)
                    frag_n += n
                    ef = metrics.fragment_errors(meta[c]["ref"], finals, "pt")
                    eb = metrics.fragment_errors(meta[c]["ref"], base, "pt")
                    e1 = metrics.fragment_errors(meta[c]["ref"], ta, "pt")
                    e2 = metrics.fragment_errors(meta[c]["ref"], tb, "pt")
                    for i in flagged:
                        d = ef[i]["errors"] - eb[i]["errors"]
                        imp += d < 0
                        wor += d > 0
                        same += d == 0
                    orac_all_e += sum(min(x["errors"], y["errors"]) for x, y in zip(e1, e2))
                    orac_gate_e += sum(min(e1[i]["errors"], e2[i]["errors"]) if i in flagged else e2[i]["errors"] for i in range(n))
                    base_e += sum(x["errors"] for x in e2)
                wers = wer_set(finals_by_clip)
                gen_secs = sum(data[(c, g1)][1] + data[(c, g2)][1] for c, _i, _s in clips)
                words_all = sum(meta[c]["words"] for c in meta)
                results["pairs"].append({
                    "g1": g1, "g2": g2, "reviewer": reviewer, "wer": wers, "rtf": round((gen_secs + rev_secs) / AUD, 3),
                    "flagged": flagged_n, "fragments": frag_n, "improved": imp, "worsened": wor, "neutral": same,
                    "oracle_all": round(orac_all_e / words_all, 4), "oracle_gate": round(orac_gate_e / words_all, 4),
                    "g2_alone": round(base_e / words_all, 4)})
                p = results["pairs"][-1]
                print(f"   {label(g1):7}+{label(g2):9} WER {wers['total']*100:5.2f}%  (rev {flagged_n}/{frag_n}, +{imp}/-{wor})  RTF {p['rtf']:.3f}", flush=True)
                with open(rev_path, "w", encoding="utf-8") as f:
                    json.dump(cache_rev, f, ensure_ascii=False)
                with open(os.path.join(args.out, "matriz.json"), "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.out, "matriz.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Resultado em", os.path.join(args.out, "matriz.json"), flush=True)


if __name__ == "__main__":
    main()
