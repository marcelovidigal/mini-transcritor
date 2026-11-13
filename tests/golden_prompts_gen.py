"""Gera tests/golden_prompts.json com os prompts renderizados para um conjunto fixo de argumentos.

Rode ANTES de mudar os textos de propósito; o teste test_prompts.py compara a renderização atual com este arquivo."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transcritor import ollama_client as oc  # noqa: E402

CTX = "Odysseus, Ítaca; fala sobre uma viagem"
cases = {
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
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_prompts.json")
json.dump(cases, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(cases), "prompts congelados em", out)
