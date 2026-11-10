"""Ambiente limpo para subprocessos (ollama, ffmpeg, ffprobe).

O PyInstaller antepõe a pasta `_internal` ao PATH (Windows) e ao LD_LIBRARY_PATH / DYLD_LIBRARY_PATH (Linux/macOS).
Programas externos herdariam isso e poderiam carregar DLLs/libs erradas do nosso pacote.
"""
import os
import sys


def clean_env() -> dict:
    env = os.environ.copy()
    bundle = getattr(sys, "_MEIPASS", None)
    if not bundle:
        return env
    bundle = os.path.normcase(os.path.abspath(bundle))
    parts = [p for p in env.get("PATH", "").split(os.pathsep)
             if p and not os.path.normcase(os.path.abspath(p)).startswith(bundle)]
    env["PATH"] = os.pathsep.join(parts)
    for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        orig = env.pop(var + "_ORIG", None)  # valor original salvo pelo bootloader do PyInstaller
        if orig is not None:
            env[var] = orig
        else:
            env.pop(var, None)
    return env
