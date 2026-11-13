"""Juiz da v2: âncora B, silêncio, confiança do A medida no arquivo, decisão por ponto e correções (sem GPU)."""
from transcritor import judge_v2 as j


def test_word_diffs_and_compose():
    a, b = "the cat sat on a mat today", "the cat sat on the mat"
    d = j.word_diffs(a, b)
    assert d == [{"a": "a", "b": "the"}, {"a": "today", "b": ""}]
    assert j.compose(a, b, ["B", "B"]) == b
    assert j.compose(a, b, ["A", "B"]) == "the cat sat on a mat"
    assert j.compose(a, b, ["A", "A"]) == a
    assert j.word_diffs("igual aqui", "Igual, aqui!") == []  # pontuação e caixa não são divergência


def test_file_signals_separate_clean_speech_from_a_that_invents_text():
    clean = j.file_signals(["um dois três", "quatro cinco"], ["um dois três", "quatro cinco"])
    assert j.a_trusted(clean) and clean["only_a_rate"] == 0
    film = j.file_signals(["fala um", "texto inventado aqui agora", "música letra"], ["fala um", "", ""])
    assert film["only_a_rate"] > 0.5 and not j.a_trusted(film)
    verbose = j.file_signals(["a b c d e f g h i j k l"], ["a b c d e f g h i j"])  # A escreve 20 % mais que o B
    assert verbose["ratio_a_b"] > j.RATIO_LOW and not j.a_trusted(verbose)


def test_silence_is_information():
    ok = {"speech": True, "confidence": 0.9}
    assert j.decide("texto do A", "", None, True) == ("", "silêncio")  # sem revisão: o B calou, vale o silêncio
    assert j.decide("texto do A", "", {"speech": False, "confidence": 0.9}, True) == ("", "silêncio")
    assert j.decide("texto do A", "", ok, True) == ("texto do A", "A(fala confirmada)")  # revisor ouviu fala e o A é confiável
    assert j.decide("texto do A", "", ok, False) == ("", "silêncio")  # A pouco confiável neste arquivo: nunca preenche o silêncio
    assert j.decide("texto do A", "", {"speech": True, "confidence": 0.2}, True) == ("", "silêncio")
    assert j.decide("", "", None, True) == ("", "B")
    assert j.decide("", "só o B", None, True) == ("só o B", "B")


def test_b_is_the_anchor_and_picks_are_applied_only_when_a_is_trusted():
    a, b = "the cat sat on a mat", "the cat sat on the mat"
    r = {"picks": ["A"], "confidence": 0.9}
    assert j.decide(a, b, None, True) == (b, "B")  # sem revisor: âncora
    assert j.decide(a, b, r, True) == (a, "A+B")
    assert j.decide(a, b, r, False) == (b, "B")  # A pouco confiável: ignora as escolhas de A
    assert j.decide(a, b, {"picks": ["B"], "confidence": 0.9}, True) == (b, "B")
    assert j.decide(a, b, {"picks": ["A"], "confidence": 0.3}, True) == (b, "B")  # sem confiança
    assert j.decide(a, b, {"picks": ["A", "A"], "confidence": 0.9}, True) == (b, "B")  # tamanho errado: descarta
    many_a, many_b = " ".join(f"a{i}" for i in range(20)), " ".join(f"b{i}" for i in range(20))
    assert j.decide(many_a, many_b, {"picks": ["A"] * 20, "confidence": 0.9}, True)[0] == many_b  # diverge demais: B


def test_fixes_correct_words_both_got_wrong_with_guardrails():
    b = "Segundo o Dr. Mall, alguns pacientes"
    assert j.apply_fixes(b, [{"from": "Mall", "to": "Moll"}]) == ("Segundo o Dr. Moll, alguns pacientes", 1)  # parecida, mantém a pontuação
    assert j.apply_fixes(b, [{"from": "Mall", "to": "Zeferino"}])[1] == 0  # nada parecida e fora do glossário
    assert j.apply_fixes(b, [{"from": "Mall", "to": "Zeferino"}], ["Zeferino"])[1] == 1  # glossário vale
    assert j.apply_fixes(b, [{"from": "Mall", "to": "Mall"}, {"from": "inexistente", "to": "x"}, "lixo"])[1] == 0
    assert j.apply_fixes(b, "não é lista") == (b, 0)
    r = {"picks": [], "fixes": [{"from": "Mall", "to": "Moll"}], "confidence": 0.9}
    assert j.decide(b, b, r, False) == ("Segundo o Dr. Moll, alguns pacientes", "B+correção")  # a correção vale mesmo com A não confiável
