"""Gera dist/MiniTranscritor/ (modo pasta) com PyInstaller. PyInstaller NÃO faz cross-compile: roda no SO de destino.

Uso:  python build.py [--archive] [--with-ffmpeg]

  --archive      gera dist/MiniTranscritor-v<versão>-<so>-<arq>.(zip|tar.gz) + .sha256 (artefato de release)
  --with-ffmpeg  copia ffmpeg/ffprobe do PATH para dist/MiniTranscritor/ffmpeg (builds "full" pesam ~450 MB)
"""
import hashlib
import os
import platform
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from transcritor.version import __version__  # noqa: E402

NAME = "MiniTranscritor"
SEP = ";" if sys.platform == "win32" else ":"


def target_tag() -> str:
    os_name = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    arch = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(platform.machine().lower(),
                                                                                       platform.machine().lower())
    return f"{os_name}-{arch}"


def make_archive(dist: str) -> str:
    base = os.path.join(dist, f"{NAME}-v{__version__}-{target_tag()}")
    path = shutil.make_archive(base, "zip" if sys.platform == "win32" else "gztar", dist, NAME)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    with open(path + ".sha256", "w") as f:
        f.write(f"{digest}  {os.path.basename(path)}\n")
    return path


def main() -> None:
    args = set(sys.argv[1:])
    dist = os.path.join(ROOT, "dist")
    subprocess.check_call([
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--console",
        "--name", NAME,
        "--exclude-module", "authlib", "--exclude-module", "waitress",  # só o modo servidor usa
        # motores opcionais (importados sob demanda em transcritor/engines.py): nunca entram no executável,
        # mesmo que o PyTorch esteja instalado no ambiente de build (o PyInstaller o embutiria: ~2,7 GB)
        *[arg for mod in ("torch", "torchvision", "torchaudio", "triton", "whisper", "faster_whisper", "whisperx", "ctranslate2", "pyannote",
                    "transformers", "tokenizers", "huggingface_hub", "tqdm", "scipy", "pandas", "pyarrow", "cv2", "onnxruntime", "plotly",
                    "grpc", "google")
          for arg in ("--exclude-module", mod)],
        "--add-data", f"{os.path.join(ROOT, 'templates')}{SEP}templates",
        "--add-data", f"{os.path.join(ROOT, 'static')}{SEP}static",
        "--add-data", f"{os.path.join(ROOT, 'audio_models.txt')}{SEP}.",
        "--add-data", f"{os.path.join(ROOT, 'whisper_models.txt')}{SEP}.",
        "--add-data", f"{os.path.join(ROOT, 'prompts')}{SEP}prompts",
        "--distpath", dist, "--workpath", os.path.join(ROOT, "build"), "--specpath", os.path.join(ROOT, "build"),
        os.path.join(ROOT, "launcher.py"),
    ], cwd=ROOT)

    out = os.path.join(dist, NAME)
    for catalog_file in ("audio_models.txt", "whisper_models.txt"):
        shutil.copy(os.path.join(ROOT, catalog_file), out)  # editáveis, ao lado do executável
    for extra in ("README.md", "docs", "prompts"):  # prompts/: editáveis, ao lado do executável (transcritor/prompts.py lê de lá)
        src = os.path.join(ROOT, extra)
        dst = os.path.join(out, extra)
        shutil.copytree(src, dst, dirs_exist_ok=True) if os.path.isdir(src) else shutil.copy(src, dst)

    if "--with-ffmpeg" in args:
        os.makedirs(os.path.join(out, "ffmpeg"), exist_ok=True)
        for tool in ("ffmpeg", "ffprobe"):
            path = shutil.which(tool)
            if not path:
                sys.exit(f"{tool} não encontrado no PATH.")
            shutil.copy(path, os.path.join(out, "ffmpeg"))

    print(f"\nPronto: {out}")
    if "--archive" in args:
        print(f"Artefato: {make_archive(dist)}")


if __name__ == "__main__":
    main()
