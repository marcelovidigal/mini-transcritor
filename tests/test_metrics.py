import re

import pytest

from transcritor import metrics, pipeline, srt
from transcritor.srt import Cue


def test_wer_basics_ignore_case_accents_and_punctuation():
    assert metrics.wer("Olá, mundo!", "ola mundo") == 0
    assert metrics.wer("a b c d", "a x c") == pytest.approx(0.5)  # 1 substituição + 1 remoção em 4 palavras
    assert metrics.wer("", "") == 0 and metrics.wer("", "algo") == 1
    assert metrics.wer("um dois", "um dois tres quatro") == 1  # inserções contam


def test_parse_srt_roundtrip():
    cues = [Cue(0, 1.5, "linha um\nlinha dois"), Cue(2.25, 3, "outra")]
    back = metrics.parse_srt(srt.to_srt(cues))
    assert [(c.start, c.end, c.text) for c in back] == [(0, 1.5, "linha um\nlinha dois"), (2.25, 3, "outra")]
    assert metrics.plain_text(back) == "linha um linha dois outra"
    assert metrics.parse_srt("") == []


def test_rule_stats_counts_each_violation():
    cues = [
        Cue(0, 4, "texto normal"),
        Cue(5, 5.3, "x" * 30),  # curta + rápida
        Cue(6, 14, "demora demais"),  # longa
        Cue(14.01, 16, "a" * 50),  # linha > 42, intervalo pequeno
        Cue(15.5, 17, "sobrepõe"),  # sobreposição com a anterior
    ]
    st = metrics.rule_stats(cues)
    assert st["cues"] == 5 and st["too_short"] == 1 and st["too_long"] == 1 and st["line_too_long"] == 1
    assert st["over_cps"] >= 2 and st["overlap"] == 1 and st["gap_too_small"] == 1


def test_clean_output_of_pipeline_style_has_no_violations():
    cues = srt.apply_style([Cue(0, 3, "Hoje vamos conversar sobre como organizar o seu tempo durante a semana"),
                            Cue(3, 6, "Primeiro, o contexto histórico do assunto.")])
    st = metrics.rule_stats(cues)
    assert st["line_too_long"] == st["too_many_lines"] == st["overlap"] == st["gap_too_small"] == 0


def test_timing_error_matches_by_words():
    ref = [Cue(10, 12, "vamos começar a aula"), Cue(20, 22, "segunda parte do assunto")]
    hyp = [Cue(10.4, 12, "vamos começar a aula"), Cue(21.0, 23, "segunda parte do assunto"), Cue(40, 41, "ruído")]
    t = metrics.timing_error(ref, hyp)
    assert t["median"] == pytest.approx(0.7) and t["unmatched"] == 0
    assert metrics.timing_error(ref, [Cue(100, 101, "outra coisa")])["unmatched"] == 1.0


def test_timing_summary_and_job_timings(client_factory, media_dir, tmp_path):
    import shutil
    from .test_api import start, wait_done

    assert "RTF 0.10" in pipeline.timing_summary({"transcricao": 6.0, "vad": 0.0}, 60) and pipeline.timing_summary({}, 60) == ""
    src = tmp_path / "t.mp3"
    shutil.copy(media_dir / "a.mp3", src)
    c, _ = client_factory()
    s, _ = wait_done(c, start(c, src).get_json()["id"])
    assert s["status"] == "done" and s["audio_seconds"] > 40
    assert {"modelo", "extracao", "vad", "transcricao", "traducao", "montagem", "total"} <= set(s["timings"])
    assert re.fullmatch(r"Tempo de processamento: \d\d:\d\d:\d\d", s["message"])
    assert pipeline.format_hms(4379) == "01:12:59" and pipeline.format_hms(0.4) == "00:00:00" and pipeline.format_hms(59.6) == "00:01:00"


def test_divergence_flags_fragments_where_two_generations_disagree():
    a = ["olá mundo como vai", "tudo bem por aqui", "", "sem texto em B"]
    b = ["olá mundo como vai", "tudo mal por lá", "", ""]
    d = metrics.divergence(a, b, threshold=0.15)
    assert d["fragments"] == 4 and d["indexes"] == [1, 3] and d["divergent"] == 2 and d["divergent_fraction"] == 0.5
    assert metrics.divergence(a, a)["divergent"] == 0 and metrics.divergence([], [])["divergent_fraction"] == 0.0
    assert metrics.divergence(["a b c d"], ["a b c x"], threshold=0.3)["divergent"] == 0  # 25% de diferença: abaixo do limiar


def test_fragment_errors_attributes_each_error_to_its_fragment():
    ref = "um dois tres quatro cinco seis sete oito"
    frags = ["um dois tres quatro", "cinco XX sete", "oito"]  # erro de substituição só no fragmento 1
    fe = metrics.fragment_errors(ref, frags)
    assert [f["errors"] for f in fe] == [0, 1, 0] and sum(f["ref_words"] for f in fe) == 8
    fe = metrics.fragment_errors("a b c d", ["a b", "d"])  # 'c' foi removida: erro vai ao fragmento vizinho
    assert sum(f["errors"] for f in fe) == 1 and sum(f["ref_words"] for f in fe) == 4
    assert metrics.fragment_errors("a b", ["a b"]) == [{"errors": 0, "ref_words": 2}]


def test_gate_analysis_precision_recall_and_oracles():
    a = [{"errors": 0, "ref_words": 10}, {"errors": 2, "ref_words": 10}, {"errors": 1, "ref_words": 10}, {"errors": 0, "ref_words": 10}]
    b = [{"errors": 0, "ref_words": 10}, {"errors": 0, "ref_words": 10}, {"errors": 3, "ref_words": 10}, {"errors": 1, "ref_words": 10}]
    g = metrics.gate_analysis(a, b, flagged=[1, 3])
    assert g["precision"] == 0.5 and g["recall"] == 0.5 and g["wer_a"] == 0.075 and g["wer_b"] == 0.1
    assert g["oracle_all"] == 0.025 and g["oracle_gated"] == 0.025  # fragmento 2 não foi marcado: A(1) fica
    assert metrics.gate_analysis(a, b, flagged=[])["oracle_gated"] == 0.075


def test_spell_numbers_for_wer_pt_and_en():
    assert metrics.spell_numbers("Trabalhe por 25 minutos e 30%", "pt-BR") == "Trabalhe por vinte e cinco minutos e trinta por cento"
    assert metrics.spell_numbers("100 e 101 e 342", "pt") == "cem e cento e um e trezentos e quarenta e dois"
    assert metrics.spell_numbers("at 9 in the morning, 45 percent", "en") == "at nine in the morning, forty five percent"
    assert metrics.spell_numbers("1999 fica", "pt") == "1999 fica" and metrics.spell_numbers("12", "ja") == "12"
    assert metrics.wer("vinte e cinco minutos", "25 minutos") > 0 and metrics.wer("vinte e cinco minutos", "25 minutos", "pt-BR") == 0
    assert metrics.wer("às nove da manhã", "às 9 da manhã", "pt") == 0


def test_time_coverage_is_language_independent():
    ref = [Cue(0, 4, "a"), Cue(6, 10, "b")]  # 8 s de fala
    perfect = metrics.coverage(ref, [Cue(0, 4, "x"), Cue(6, 10, "y")])
    assert perfect == {"recall": 1.0, "precision": 1.0, "iou": 1.0}
    half = metrics.coverage(ref, [Cue(0, 2, "x"), Cue(12, 14, "y")])  # cobre 2 de 8 s; 2 de 4 s da hipótese estão sobre fala
    assert half["recall"] == 0.25 and half["precision"] == 0.5 and 0.1 < half["iou"] < 0.3
    assert metrics.coverage(ref, []) == {"recall": 0.0, "precision": 0.0, "iou": 0.0}
    overlapping = metrics.coverage([Cue(0, 5, "a"), Cue(3, 8, "b")], [Cue(0, 8, "x")])  # a referência se sobrepõe: união de 8 s
    assert overlapping["recall"] == 1.0 and overlapping["precision"] == 1.0


def test_chrf_rewards_similar_translations_and_ignores_accents_and_case():
    assert metrics.chrf("O gato subiu no telhado.", "o gato subiu no telhado") == 1.0
    close = metrics.chrf("O gato subiu no telhado ontem.", "O gato subiu no telhado hoje.")
    far = metrics.chrf("O gato subiu no telhado ontem.", "Vamos jantar fora esta noite com amigos.")
    assert close > 0.6 > far and metrics.chrf("", "texto") == 0.0


def test_error_breakdown_separates_substitutions_omissions_and_insertions():
    ref = "um dois tres quatro cinco"
    r = metrics.error_breakdown(ref, "um dois tres quatro cinco", "pt")
    assert r["wer"] == 0 and r["ins"] == 0
    r = metrics.error_breakdown(ref, "um dois tres quatro cinco seis sete", "pt")  # texto a mais (inventado)
    assert r["ins"] == 0.4 and r["sub"] == 0 and r["del"] == 0 and r["hyp_words"] == 7
    r = metrics.error_breakdown(ref, "um dois quatro", "pt")  # fala perdida
    assert r["del"] == 0.4 and r["ins"] == 0
    r = metrics.error_breakdown(ref, "um dois xxx quatro cinco", "pt")
    assert r["sub"] == 0.2 and r["wer"] == 0.2
    assert metrics.error_breakdown("", "algo", "pt")["wer"] == 1.0
