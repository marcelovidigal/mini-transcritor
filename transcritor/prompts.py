"""Prompts dos agentes, lidos da pasta `prompts/` (raiz do projeto; ao lado do executável no app empacotado).

Cada arquivo é um modelo de texto com campos `{nome}`; `render` troca só os campos informados (em uma única passada, então um valor
que contenha `{algo}` nunca é expandido de novo). O ÚLTIMO fim de linha de cada arquivo é ignorado; espaços no começo e no fim
da linha contam (alguns trechos começam com um espaço de propósito, para emendar no texto anterior).

Para experimentar outros prompts sem mexer no projeto: defina MT_PROMPTS_DIR com uma pasta com os mesmos nomes de arquivo.
Veja prompts/README.md."""
import os
import re

from .ffmpeg_manager import app_dir

_FIELD = re.compile(r"\{(\w+)\}")
_cache: dict[tuple[str, str], str] = {}


class PromptError(Exception):
    """Arquivo de prompt ausente ou ilegível."""


def prompts_dir() -> str:
    return os.environ.get("MT_PROMPTS_DIR") or os.path.join(app_dir(), "prompts")


def load(name: str) -> str:
    base = prompts_dir()
    key = (base, name)
    if key not in _cache:
        path = os.path.join(base, name + ".md")
        try:
            with open(path, encoding="utf-8", newline="") as f:
                text = f.read().replace("\r\n", "\n")
        except OSError as e:
            raise PromptError(f"Prompt '{name}' não encontrado em {base} ({e}). A pasta prompts/ faz parte do app.")
        _cache[key] = text[:-1] if text.endswith("\n") else text
    return _cache[key]


def render(name: str, **values) -> str:
    return _FIELD.sub(lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), load(name))


def clear_cache() -> None:
    _cache.clear()
