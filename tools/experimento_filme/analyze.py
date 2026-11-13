"""Análise consolidada do experimento do longa-metragem, refeita a partir das legendas salvas (saidas/).

CORREÇÃO: a legenda de referência extraída do filme ia até 5648 s (o corte `-t` não vale para legendas), 3x mais que o trecho
de 30 min; por isso os WER calculados durante a execução estavam inflados. Aqui a referência é recortada à janela do trecho e os
tempos das saídas são corrigidos pelo deslocamento do corte no keyframe (offset.json)."""
import difflib
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # raiz do projeto
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from transcritor import metrics, srt  # noqa: E402

base = os.path.dirname(os.path.abspath(__file__))
SYSTEMS = ["e4b", "turbo", "whisperx_diar", "qualidade"]
WINDOW_END = 1800.0


def parse(p):
    return metrics.parse_srt(open(os.path.join(base, p), encoding="utf-8").read())


ref_all = parse("ref_limpa.srt")
keep = [i for i, c in enumerate(ref_all) if c.start < WINDOW_END]
ref = [ref_all[i] for i in keep]
ref_pt = {"e4b": [c for i, c in enumerate(parse("ref_ptbr.srt")) if i in set(keep)],
          "12b": [c for i, c in enumerate(parse("ref_ptbr_12b.srt")) if i in set(keep)]}
for k, v in ref_pt.items():
    assert len(v) == len(ref), (k, len(v), len(ref))
ref_text = " ".join(c.text for c in ref)
words_ref = len(metrics.normalize(ref_text))
print(f"referência recortada: {len(ref)} legendas, {words_ref} palavras (a original ia até {ref_all[-1].end:.0f} s)")
json.dump({"cues": len(ref), "words": words_ref}, open(os.path.join(base, "ref_stats.json"), "w"))
for name, cues in (("ref_trim.srt", ref),):
    open(os.path.join(base, name), "w", encoding="utf-8").write(srt.to_srt(cues))


def hw(cues):
    return len(metrics.normalize(" ".join(c.text for c in cues)))


def shifted(cues, off):
    out = []
    for c in cues:
        s, e = c.start - off, c.end - off
        if s < 0 or s >= WINDOW_END:  # fora da janela coberta pela referência
            continue
        out.append(srt.Cue(s, e, c.text))
    return out


def estimate_offset():
    src = parse("saidas/segmento_30min.whisperx_diar.src.srt")
    hws = [set(metrics.normalize(h.text)) for h in src]
    diffs = []
    for r in ref:
        rw = set(metrics.normalize(r.text))
        if len(rw) < 3:
            continue
        best, bs = None, 0.0
        for h, w in zip(src, hws):
            if not w or abs(h.start - (r.start + 7.2)) > 12:
                continue
            sc = len(rw & w) / len(rw | w)
            if sc > bs:
                best, bs = h, sc
        if best is not None and bs >= 0.6:
            diffs.append(best.start - r.start)
    return statistics.median(diffs), len(diffs)


OFF, n_off = estimate_offset()
print(f"deslocamento (corte no keyframe): {OFF:.2f} s (WhisperX, {n_off} pares de falas)")
json.dump({"offset_s": round(OFF, 2), "pairs": n_off}, open(os.path.join(base, "offset.json"), "w"))


def align_scores(hyp_text, ref_text_, lang):
    r, h = metrics.normalize(ref_text_, lang), metrics.normalize(hyp_text, lang)
    sm = difflib.SequenceMatcher(None, r, h, autojunk=False)
    match = sum(b.size for b in sm.get_matching_blocks())
    prec, rec = match / max(len(h), 1), match / max(len(r), 1)
    return {"wer": metrics.wer(" ".join(r), " ".join(h), None), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "hyp_words": len(h), "ref_words": len(r)}


def timing(src_cues, jacc=0.5, window=10.0):
    hws = [set(metrics.normalize(h.text)) for h in src_cues]
    errs, matched, usable = [], 0, 0
    for r in ref:
        rw = set(metrics.normalize(r.text))
        if len(rw) < 3:
            continue
        usable += 1
        best, bs = None, 0.0
        for h, w in zip(src_cues, hws):
            if not w or abs(h.start - r.start) > window:
                continue
            sc = len(rw & w) / len(rw | w)
            if sc > bs:
                best, bs = h, sc
        if best is not None and bs >= jacc:
            matched += 1
            errs.append(abs(best.start - r.start))
    errs.sort()
    return {"matched": round(matched / usable, 3), "median_s": round(statistics.median(errs), 2) if errs else None,
            "p90_s": round(errs[int(0.9 * (len(errs) - 1))], 2) if errs else None,
            "within_1s": round(sum(1 for e in errs if e <= 1.0) / usable, 3), "usable_cues": usable}


out = {}
for name in SYSTEMS:
    res = json.load(open(os.path.join(base, f"res_{name}.json"), encoding="utf-8"))[0]
    src = shifted(parse(f"saidas/segmento_30min.{name}.src.srt"), OFF)
    fin_all = parse(f"saidas/segmento_30min.{name}.srt")
    fin = shifted(fin_all, OFF)
    src_text, fin_text = " ".join(c.text for c in src), " ".join(c.text.replace("\n", " ") for c in fin)
    a_src = align_scores(src_text, ref_text, "en")
    pt = {}
    for k in ("e4b", "12b"):
        rt = " ".join(c.text for c in ref_pt[k])
        pt[k] = {**align_scores(fin_text, rt, "pt"), "chrf": round(metrics.chrf(rt, fin_text), 4)}
    cov = metrics.coverage(ref, fin)
    rules = metrics.rule_stats(fin_all)
    out[name] = {
        "src": a_src, "timing": timing(src), "pt": pt, "coverage": cov, "rules": rules,
        "cues": len(fin_all), "words_out_pt": len(metrics.normalize(" ".join(c.text for c in fin_all))),
        "wall_s": res["wall_s"], "rtf": res["rtf"], "stages_s": res["stages_s"], "warnings": res["warnings"],
        "speakers": res.get("speakers"), "turns": res.get("turns"), "report": res.get("report")}
    print(f"{name:14} src: WER {a_src['wer']*100:5.1f}% prec {a_src['precision']*100:5.1f}% rec {a_src['recall']*100:5.1f}% | tempo: casadas {out[name]['timing']['matched']:.0%} "
          f"mediana {out[name]['timing']['median_s']} s p90 {out[name]['timing']['p90_s']} s | pt vs ref(e4b): WER {pt['e4b']['wer']*100:5.1f}% chrF {pt['e4b']['chrf']:.3f} | "
          f"pt vs ref(12b): WER {pt['12b']['wer']*100:5.1f}% chrF {pt['12b']['chrf']:.3f} | cobertura rec {cov['recall']:.0%} prec {cov['precision']:.0%}", flush=True)

# concordância entre os sistemas (texto na língua falada)
agree = {}
texts = {n: " ".join(c.text for c in shifted(parse(f"saidas/segmento_30min.{n}.src.srt"), OFF)) for n in SYSTEMS}
for a in SYSTEMS:
    for b in SYSTEMS:
        if a < b:
            agree[f"{a}~{b}"] = round(metrics.wer(texts[a], texts[b], "en"), 4)
out["_agreement_src_wer"] = agree
out["_reference"] = {"cues": len(ref), "words": words_ref, "offset_s": round(OFF, 2)}
json.dump(out, open(os.path.join(base, "analise.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("concordância (WER entre os textos em inglês):", agree)
