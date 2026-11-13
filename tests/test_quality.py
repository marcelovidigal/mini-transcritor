from transcritor import quality


def test_collapse_repeats_only_for_loops():
    assert quality.collapse_repeats("e e e e e e e depois fui") == "e depois fui"
    assert quality.collapse_repeats("não sei não sei não sei não sei mesmo") == "não sei mesmo"
    assert quality.collapse_repeats("não não não, eu sei") == "não não não, eu sei"  # 3 vezes é fala normal
    assert quality.collapse_repeats("texto normal sem laço") == "texto normal sem laço"


def test_ghost_phrases_only_when_they_are_the_whole_text():
    for g in ("Legendas pela comunidade Amara.org", "Obrigado por assistir!", "Thanks for watching.", "Subtitles by the Amara.org community"):
        assert quality.is_ghost(g), g
    assert not quality.is_ghost("Obrigado por assistir ao jogo comigo ontem, foi ótimo")
    assert not quality.is_ghost("As legendas pela primeira vez ficaram boas, disse ele")


def test_seam_repeat_is_removed_only_for_4_plus_words():
    assert quality.strip_seam_repeat("e foi assim que tudo começou", "foi assim que tudo começou mesmo") == "mesmo"
    assert quality.strip_seam_repeat("eu sei", "eu sei que não") == "eu sei que não"  # curto demais: pode ser legítimo
    assert quality.strip_seam_repeat("a casa é grande", "outra frase qualquer aqui") == "outra frase qualquer aqui"


def test_clean_texts_reports_what_it_did():
    texts = ["Primeira frase normal do vídeo.", "Legendas pela comunidade Amara.org", "x" * 400, "Segunda parte e e e e e e fim.",
             "parte e e e e e e fim. Depois continua", ""]
    out, rep = quality.clean_texts(texts, [10, 10, 5, 10, 10, 10])
    assert out[0] == texts[0] and out[1] == "" and out[2] == "" and out[5] == ""
    assert "e e e" not in out[3]
    assert rep.ghosts == 1 and rep.too_fast == 1 and rep.collapsed >= 1
    assert len(rep.messages()) >= 3


def test_clean_texts_leaves_good_text_untouched():
    texts = ["Hoje vamos falar de um assunto importante.", "Primeiro, o contexto histórico."]
    out, rep = quality.clean_texts(texts, [8, 6])
    assert out == texts and rep.messages() == []


# ---------- contexto/glossário e limpeza no pipeline ----------

def test_context_goes_into_prompts_only_when_given():
    from transcritor.ollama_client import build_batch_translate_prompt, build_prompt, build_translate_prompt

    assert "Context" not in build_prompt() and "Context" not in build_translate_prompt("en", "x")
    assert "fibrilação atrial" in build_prompt("Aula de cardiologia; fibrilação atrial")
    assert "Dra. Helena Duarte" in build_translate_prompt("en", "x", "Dra. Helena Duarte")
    assert "Dra. Helena Duarte" in build_batch_translate_prompt("en", ["x"], "Dra. Helena Duarte")
    assert len(build_prompt("a" * 2000)) < len(build_prompt()) + 700  # limitado a 500 caracteres


def test_job_context_reaches_ollama_client_and_engine(client_factory, media_dir, tmp_path):
    import shutil
    from .test_api import start, wait_done

    src = tmp_path / "c.mp3"
    shutil.copy(media_dir / "a.mp3", src)
    c, fake = client_factory()
    seen = []
    orig = fake.transcribe
    fake.transcribe = lambda model, wav, context="": (seen.append(context), orig(model, wav))[1]
    s, _ = wait_done(c, start(c, src, context="Aula de cardiologia").get_json()["id"])
    assert s["status"] == "done" and set(seen) == {"Aula de cardiologia"}
    assert start(c, src, context="x" * 501).status_code == 400


def test_pipeline_drops_ghost_chunks_with_a_warning(client_factory, media_dir, tmp_path):
    import shutil
    from .test_api import start, wait_done

    src = tmp_path / "g.mp3"
    shutil.copy(media_dir / "a.mp3", src)
    c, fake = client_factory()
    fake.transcribe = lambda model, wav: "Legendas pela comunidade Amara.org"
    s, _ = wait_done(c, start(c, src).get_json()["id"])
    assert s["status"] == "error" and "nenhum trecho" in s["error"]
    assert any("inventadas" in w for w in s["warnings"])
