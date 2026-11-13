from transcritor import srt
from transcritor.srt import Cue


def lines(text):
    return text.split("\n")


def test_short_text_stays_on_one_line():
    assert srt.wrap_lines("Olá, tudo bem com você?") == "Olá, tudo bem com você?"


def test_long_text_wraps_in_two_lines_up_to_42_chars():
    out = srt.wrap_lines("Hoje vamos conversar sobre como organizar o seu tempo durante a semana")
    ls = lines(out)
    assert len(ls) == 2 and all(len(x) <= srt.MAX_LINE_CHARS for x in ls)
    assert len(ls[0]) <= len(ls[1]) + 8  # pirâmide: linha de baixo maior (ou quase igual)


def test_wrap_never_ends_top_line_with_article_or_preposition():
    out = srt.wrap_lines("Eu preciso falar com o diretor sobre a reunião de amanhã com todos os clientes")
    first = lines(out)[0].split()[-1].lower()
    assert first not in {"o", "a", "de", "com", "os", "da", "do", "e", "que"}


def test_wrap_prefers_breaking_after_punctuation_and_avoids_orphans():
    out = srt.wrap_lines("Quando cheguei em casa, percebi que tinha esquecido as chaves")
    assert lines(out)[0].endswith("casa,")
    out = srt.wrap_lines("Sim aqui está o relatório completo que você pediu ontem à tarde")
    assert len(lines(out)[0].split()) >= 3  # nunca 1–2 palavras na linha de cima


def test_text_longer_than_two_lines_is_still_cut_at_the_limit():
    out = srt.wrap_lines("palavra " * 15)
    assert len(lines(out)) == 2 and len(lines(out)[0]) <= srt.MAX_LINE_CHARS


def test_style_extends_end_for_reading_speed_when_there_is_room():
    cues = srt.apply_style([Cue(0.0, 1.0, "x" * 17 + " " + "y" * 7), Cue(3.0, 5.0, "ok")])
    assert cues[0].end - cues[0].start >= 1.4 - 1e-9 and cues[0].end <= 1.0 + srt.MAX_EXTEND + 1e-9


def test_style_enforces_min_gap_between_contiguous_cues():
    cues = srt.apply_style([Cue(0.0, 3.0, "primeira legenda aqui"), Cue(3.0, 6.0, "segunda legenda aqui")])
    assert cues[1].start - cues[0].end >= srt.MIN_GAP - 1e-9


def test_style_caps_duration_and_never_overlaps():
    cues = srt.apply_style([Cue(0.0, 12.0, "texto curto"), Cue(12.0, 14.0, "outro")])
    assert cues[0].end - cues[0].start <= srt.MAX_DURATION + 1e-9
    assert all(a.end <= b.start for a, b in zip(cues, cues[1:]))


def test_style_min_duration_for_very_short_cue():
    cues = srt.apply_style([Cue(0.0, 0.3, "Oi"), Cue(10.0, 11.0, "tchau")])
    assert cues[0].end - cues[0].start >= 0.3 + srt.MAX_EXTEND - 1e-9 or cues[0].end - cues[0].start >= srt.MIN_DURATION


def test_fast_reading_counts_only_cues_above_the_limit():
    ok = Cue(0, 4, "a" * 40)  # 10 cps
    fast = Cue(5, 6, "b" * 40)  # 40 cps
    assert srt.fast_reading([ok, fast]) == 1


def test_srt_output_keeps_two_line_cues_valid():
    cues = srt.apply_style([Cue(0.0, 4.0, "Hoje vamos conversar sobre como organizar o seu tempo durante a semana")])
    out = srt.to_srt(cues)
    assert out.startswith("1\n00:00:00,000 --> ") and out.count("\n") >= 4
    assert out.strip().count("\n") == 3  # número, tempos, 2 linhas de texto (uma só legenda)


def test_every_cue_fits_two_lines_of_42_chars():
    text = ("Escolha apenas um hábito para praticar durante duas semanas e só depois acrescente o próximo. "
            "Hoje vamos conversar sobre três hábitos simples que podem melhorar muito a sua rotina de estudos e de trabalho.")
    cues = srt.cues_for_chunk(0, 20, text)
    for c in cues:
        ls = srt.wrap_lines(c.text).split("\n")
        assert len(ls) <= 2 and all(len(x) <= srt.MAX_LINE_CHARS for x in ls), c.text


def test_long_sentence_is_split_in_balanced_parts_not_84_plus_a_stub():
    sent = ("Hoje vamos conversar sobre três hábitos simples que podem melhorar muito a sua rotina de estudos "
            "e de trabalho durante toda a semana")
    parts = srt.split_text(sent)
    assert len(parts) == 2 and abs(len(parts[0]) - len(parts[1])) <= 25  # ex.: 84 + 36 seria desequilibrado
    assert parts[0].split()[-1].lower() not in {"a", "o", "de", "e", "que", "sua", "seu"}  # não termina em artigo/preposição


def test_short_sentences_are_still_grouped():
    parts = srt.split_text("Sim. Claro. Pode ser. Vamos começar agora, então.")
    assert 1 <= len(parts) <= 2 and " ".join(parts).replace("  ", " ") == "Sim. Claro. Pode ser. Vamos começar agora, então."
