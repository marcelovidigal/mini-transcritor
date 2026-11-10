"""Smoke test do executável gerado: sobe sem preparo (sem Ollama/FFmpeg) e confere página, estáticos e API.

Uso:  python smoke_test.py [caminho/para/MiniTranscritor(.exe)]
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(ROOT, "dist", "MiniTranscritor", "MiniTranscritor" + (".exe" if sys.platform == "win32" else ""))
PORT = 5099


def get(path):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=5) as r:
        return r.status, r.read()


def main() -> int:
    exe = sys.argv[1] if len(sys.argv) > 1 else EXE
    if not os.path.isfile(exe):
        print(f"Executável não encontrado: {exe}")
        return 2
    env = dict(os.environ, PORT=str(PORT), NO_BROWSER="1", MINITRANSCRITOR_NO_BOOTSTRAP="1",
               MINITRANSCRITOR_HOME=tempfile.mkdtemp(prefix="mt_smoke_"))
    proc = subprocess.Popen([exe], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                get("/")
                break
            except OSError:
                if proc.poll() is not None:
                    print("O executável encerrou sozinho.")
                    return 1
                time.sleep(1)
        else:
            print("Servidor não respondeu em 60 s.")
            return 1
        checks = {
            "página": b"Mini Transcritor" in get("/")[1],
            "bootstrap css": get("/static/vendor/bootstrap.min.css")[0] == 200,
            "app.js": get("/static/js/app.js")[0] == 200,
            "idiomas": len(json.loads(get("/api/languages")[1])) > 3,
            "setup": json.loads(get("/api/setup")[1])["status"] == "ready",
            "configurações": json.loads(get("/api/settings")[1])["base_model"] == "gemma4:e4b",
            "api de modelos responde": "ok" in json.loads(get("/api/models")[1]),  # sem Ollama: ok=false + erro
        }
        for name, ok in checks.items():
            print(("OK   " if ok else "FALHA"), name)
        return 0 if all(checks.values()) else 1
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
