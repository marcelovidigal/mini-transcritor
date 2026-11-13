"""Catálogo estático de modelos com áudio (audio_models.txt) e mesclagem com os instalados."""
import os
import sys

from .ffmpeg_manager import app_dir

FILE_NAME = "audio_models.txt"
_FALLBACK = [
    {"name": "gemma4:e4b", "size": "6.6 GB", "note": "Padrão recomendado."},
    {"name": "gemma4:e2b", "size": "4.6 GB", "note": "Mais leve e rápido."},
    {"name": "gemma4:12b", "size": "8.0 GB", "note": "Maior."},
]


def norm(name: str) -> str:
    """Normaliza o nome do modelo: 'nemotron3' == 'nemotron3:latest'."""
    name = (name or "").strip().lower()
    return name if ":" in name else name + ":latest"


def has_explicit_tag(name: str) -> bool:
    """True se o nome tem tag e ela não é `latest` (ex.: 'gemma4:e4b', 'nemotron3:33b-q4_K_M')."""
    base, _, tag = (name or "").strip().partition(":")
    return bool(base) and bool(tag) and tag.lower() != "latest"


def _candidate_paths() -> list[str]:
    paths = [os.path.join(app_dir(), FILE_NAME)]  # ao lado do exe / raiz do projeto (editável)
    if getattr(sys, "_MEIPASS", None):
        paths.append(os.path.join(sys._MEIPASS, FILE_NAME))  # cópia embutida
    return paths


def parse(text: str) -> list[dict]:
    out, seen = [], set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "!")):  # "!" = modelo escondido (parse_excluded)
            continue
        parts = [p.strip() for p in line.split("|")]
        name = parts[0]
        if not name or " " in name or norm(name) in seen or not has_explicit_tag(name):
            continue  # sem tag ou com `latest`: ignorado (regra: tag descritiva com parâmetros)
        seen.add(norm(name))
        out.append({"name": name, "size": parts[1] if len(parts) > 1 else "", "note": parts[2] if len(parts) > 2 else ""})
    return out


def parse_excluded(text: str) -> set[str]:
    """Modelos que NÃO devem aparecer nas opções mesmo instalados: linhas `!nome:tag | motivo` do arquivo de catálogo."""
    out = set()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("!"):
            name = line[1:].split("|")[0].strip()
            if name:
                out.add(norm(name))
    return out


def load_excluded(path: str | None = None) -> set[str]:
    for p in ([path] if path else _candidate_paths()):
        try:
            with open(p, encoding="utf-8") as f:
                found = parse_excluded(f.read())
            return found
        except OSError:
            continue
    return set()


def load(path: str | None = None) -> list[dict]:
    for p in ([path] if path else _candidate_paths()):
        try:
            with open(p, encoding="utf-8") as f:
                items = parse(f.read())
            if items:
                return items
        except OSError:
            continue
    return [dict(m) for m in _FALLBACK]


def merge(catalog: list[dict], installed: list[dict], excluded: set[str] | None = None) -> list[dict]:
    """Catálogo (na ordem do arquivo) + instalados fora do catálogo. Marca `installed` e `recommended`.
    Modelos em `excluded` (linhas `!nome` do catálogo) somem das opções mesmo que estejam instalados."""
    excluded = excluded or set()
    installed_by_norm = {norm(m["name"]): m for m in installed if norm(m["name"]) not in excluded}
    result, in_catalog = [], set()
    for c in catalog:
        key = norm(c["name"])
        in_catalog.add(key)
        inst = installed_by_norm.get(key)
        result.append({
            "name": inst["name"] if inst else c["name"], "installed": bool(inst), "recommended": True,
            "size": inst.get("size", 0) if inst else 0, "approx_size": c["size"], "note": c["note"],
        })
    for key, m in installed_by_norm.items():
        if key not in in_catalog:
            result.append({"name": m["name"], "installed": True, "recommended": False,
                           "size": m.get("size", 0), "approx_size": "", "note": "Instalado (fora do catálogo)"})
    # instalados primeiro (estável: mantém a prioridade do catálogo dentro de cada grupo)
    return sorted(result, key=lambda m: not m["installed"])
