"""A pasta prompts/ é a fonte dos prompts dos agentes: a renderização tem de ser IDÊNTICA à congelada em golden_prompts.json
(gerada antes da migração; se um prompt mudar de propósito, regenere com tests/golden_prompts_gen.py e explique no commit)."""
import json
import os

import pytest

from transcritor import ollama_client as oc
from transcritor import prompts

HERE = os.path.dirname(os.path.abspath(__file__))
CTX = "Odysseus, Ítaca; fala sobre uma viagem"


def current():
    return {
        "transcribe": oc.build_prompt(""),
        "transcribe_ctx": oc.build_prompt(CTX),
        "feedback_full": oc.build_feedback_note("hello world", "A misheard a name", ["Moll", "Ítaca"]),
        "feedback_prev_only": oc.build_feedback_note("hello world"),
        "feedback_empty": oc.build_feedback_note(""),
        "review_ab": oc.build_review_prompt("texto a", "texto b"),
        "review_ab_ctx": oc.build_review_prompt("texto a", "texto b", CTX),
        "review_abc": oc.build_review_prompt("texto a", "texto b", "", "texto c"),
        "review_notes": oc.build_review_prompt("texto a", "texto b", CTX, "", True),
        "review_notes_prev": oc.build_review_prompt("texto a", "texto b", CTX, "texto c", True, True),
        "polish": oc.build_polish_prompt("", ["um dois", "três quatro"], ""),
        "polish_ctx": oc.build_polish_prompt("pt-BR", ["um dois", "três quatro"], "Odysseus, Ítaca"),
        "translation_review": oc.build_translation_review_prompt("pt-BR", [("Hello", "Olá"), ("Fine", "Bem")], ""),
        "translation_review_ctx": oc.build_translation_review_prompt("pt-BR", [("Hello", "Olá")], "Odysseus, Ítaca"),
        "translate": oc.build_translate_prompt("pt-BR", "Hello there", ""),
        "translate_ctx": oc.build_translate_prompt("pt-BR", "Hello there", CTX),
        "translate_batch": oc.build_batch_translate_prompt("pt-BR", ["Hello", "How are you?"], ""),
        "translate_batch_ctx": oc.build_batch_translate_prompt("pt-BR", ["Hello", "How are you?"], CTX),
        "self_review": oc.build_self_review_prompt("texto a", 8),
        "self_review_doubts": oc.build_self_review_prompt("texto a", 8, CTX, "antes", "depois", [{"current": "gato", "other": "pato"}, {"current": "", "other": "o"}], True),
        "variant_literal": oc.build_variant_prompt("literal", CTX),
        "variant_careful": oc.build_variant_prompt("careful"),
    }


def test_prompts_from_files_are_identical_to_the_frozen_ones():
    golden = json.load(open(os.path.join(HERE, "golden_prompts.json"), encoding="utf-8"))
    now = current()
    assert set(now) == set(golden)
    for k in golden:
        assert now[k] == golden[k], k


def test_render_only_fills_given_fields_in_one_pass():
    assert prompts.render("translate", lang="X", context="", text="a {lang} b") == prompts.render("translate", lang="X", context="", text="a {lang} b")
    out = prompts.render("translate", lang="X", context="", text="a {lang} b")
    assert out.endswith("Text:\na {lang} b")  # valor com {campo} não é expandido de novo
    assert "{n}" in prompts.render("polish", language="x")  # campo não informado fica como está


def test_override_dir_and_missing_prompt(tmp_path, monkeypatch):
    (tmp_path / "transcribe.md").write_text("Only this.{context}{feedback}\n", encoding="utf-8")
    monkeypatch.setenv("MT_PROMPTS_DIR", str(tmp_path))
    prompts.clear_cache()
    try:
        assert oc.build_prompt("") == "Only this."
        with pytest.raises(prompts.PromptError):
            prompts.load("review")
    finally:
        monkeypatch.delenv("MT_PROMPTS_DIR")
        prompts.clear_cache()


def test_every_prompt_file_is_used_by_the_code():
    """Nenhum arquivo órfão em prompts/ (cada um é lido por algum agente)."""
    root = os.path.dirname(HERE)
    src = "".join(open(os.path.join(root, "transcritor", f), encoding="utf-8").read()
                  for f in os.listdir(os.path.join(root, "transcritor")) if f.endswith(".py"))
    for f in os.listdir(os.path.join(root, "prompts")):
        if f.endswith(".md") and f != "README.md":
            assert f'"{f[:-3]}"' in src, f"prompts/{f} não é usado"
