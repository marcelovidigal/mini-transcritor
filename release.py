"""Release local: testes → build → smoke test → artefato (+ sha256) → (opcional) publicação no GitHub.

Uso:
  python release.py                 # dry-run: tudo, menos publicar (gera dist/MiniTranscritor-v…-<so>-<arq>.*)
  python release.py --publish       # idem + cria o Release vX.Y.Z no GitHub com `gh` (pede confirmação)
  python release.py --skip-tests    # pula pytest (não recomendado)

Atenção: este script gera o artefato apenas do SO em que roda. O release OFICIAL com todos os SO
(Windows, Linux x64/arm64, macOS) sai do GitHub Actions (.github/workflows/release.yml) ao criar a tag vX.Y.Z.
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from transcritor.version import __version__  # noqa: E402


def run(*cmd, **kw):
    print("\n$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true", help="cria o Release no GitHub com gh (pede confirmação)")
    ap.add_argument("--yes", action="store_true", help="não pedir confirmação ao publicar")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()

    if not re.fullmatch(r"\d+\.\d+\.\d+(-[0-9A-Za-z.]+)?", __version__):
        sys.exit(f"Versão inválida em transcritor/version.py: {__version__}")
    tag = f"v{__version__}"

    if not a.skip_tests:
        run(sys.executable, "-m", "pytest", "-q")
    run(sys.executable, "build.py", "--archive")
    run(sys.executable, "smoke_test.py")

    assets = sorted(glob.glob(os.path.join(ROOT, "dist", f"MiniTranscritor-{tag}-*.*")))
    print("\nArtefatos:\n  " + "\n  ".join(assets))
    if not a.publish:
        print("\nDry-run concluído (nada foi publicado). Use --publish, ou crie a tag para o release oficial via CI.")
        return 0

    if not shutil.which("gh"):
        sys.exit("GitHub CLI (gh) não encontrado. Instale em https://cli.github.com e faça `gh auth login`.")
    if not a.yes and input(f"Publicar o release {tag} com {len(assets)} arquivo(s) no GitHub? [s/N] ").lower() != "s":
        print("Cancelado.")
        return 1
    prerelease = ["--prerelease"] if "-" in __version__ else []
    run("gh", "release", "create", tag, *assets, "--title", f"Mini Transcritor {tag}", "--generate-notes", *prerelease)
    return 0


if __name__ == "__main__":
    sys.exit(main())
