"""Navegador de pastas do servidor (uso local): lista diretórios e arquivos de mídia."""
import os
import string
import sys

from .media import is_media_file, kind_from_extension


def _drives() -> list[str]:
    return [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]


def list_dir(path: str | None) -> dict:
    """Retorna {path, parent, dirs:[{name,path}], files:[{name,path,kind}]}. Levanta OSError se ilegível."""
    if not path:
        if sys.platform == "win32":
            return {"path": "", "parent": None, "dirs": [{"name": d, "path": d} for d in _drives()], "files": []}
        path = os.path.expanduser("~")
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise NotADirectoryError(path)
    dirs, files = [], []
    for entry in sorted(os.scandir(path), key=lambda e: e.name.lower()):
        if entry.name.startswith("."):
            continue
        try:
            if entry.is_dir():
                dirs.append({"name": entry.name, "path": entry.path})
            elif entry.is_file() and is_media_file(entry.name):
                files.append({"name": entry.name, "path": entry.path, "kind": kind_from_extension(entry.name)})
        except OSError:
            continue
    parent = os.path.dirname(path)
    if parent == path:  # raiz
        parent = "" if sys.platform == "win32" else None
    return {"path": path, "parent": parent, "dirs": dirs, "files": files}
