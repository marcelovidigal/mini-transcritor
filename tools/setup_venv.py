r"""Cria/atualiza o ambiente virtual do projeto (.venv) com TODAS as bibliotecas usadas, isolado dos outros apps.

    python tools/setup_venv.py                 # PyTorch com Intel XPU (padrão), motores, ADK e ferramentas de desenvolvimento
    python tools/setup_venv.py --torch cuda    # GPU NVIDIA (build CUDA do PyTorch)
    python tools/setup_venv.py --torch cpu     # só CPU
    python tools/setup_venv.py --torch none    # não instala o PyTorch (sem OpenAI Whisper/WhisperX)
    python tools/setup_venv.py --no-quality --no-dev

Depois, use SEMPRE o Python do .venv:   .venv\Scripts\python app.py   (Windows)   ·   .venv/bin/python app.py   (Linux/macOS)

Ordem (importa): 1) PyTorch da build certa (a do PyPI não serve para XPU); 2) requirements.txt, -engines, -quality, -dev;
3) WhisperX SEM dependências (ele fixa torch~=2.8, que trocaria o PyTorch escolhido) + as dependências dele, exceto o torch.
O .venv NÃO enxerga os pacotes do Python global (isolamento total).
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV = os.path.join(ROOT, ".venv")
PY = os.path.join(VENV, "Scripts", "python.exe") if sys.platform == "win32" else os.path.join(VENV, "bin", "python")
INDEX = {"xpu": "https://download.pytorch.org/whl/xpu", "cuda": "https://download.pytorch.org/whl/cu128",
         "cpu": "https://download.pytorch.org/whl/cpu"}
# o que o WhisperX pede além do torch (que já veio do passo 1)
WHISPERX_DEPS = ["faster-whisper", "ctranslate2", "huggingface-hub", "nltk", "omegaconf", "pandas", "pyannote-audio", "transformers"]

CHECK = """
import importlib.metadata as m, importlib.util as u
for n in ('flask', 'numpy', 'requests', 'authlib', 'waitress', 'openai-whisper', 'faster-whisper', 'whisperx', 'torch',
          'google-adk', 'pyinstaller', 'playwright'):
    try:
        print(f'{n:16} {m.version(n)}')
    except Exception:
        print(f'{n:16} (não instalado)')
try:
    import torch
    print('dispositivos:', {'cuda': torch.cuda.is_available(),
                            'xpu': bool(getattr(torch, 'xpu', None) and torch.xpu.is_available()),
                            'mps': bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())})
except Exception as e:
    print('torch:', e)
print('litellm instalado:', u.find_spec('litellm') is not None)
"""


def run(*args):
    print("\n$", " ".join(args), flush=True)
    subprocess.check_call(list(args), cwd=ROOT)


def pip(*args):
    run(PY, "-m", "pip", *args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--torch", choices=["xpu", "cuda", "cpu", "none"], default="xpu")
    ap.add_argument("--no-quality", action="store_true", help="não instala o Google ADK")
    ap.add_argument("--no-dev", action="store_true", help="não instala PyInstaller/Playwright")
    ap.add_argument("--no-whisperx", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(PY):
        run(sys.executable, "-m", "venv", VENV)  # sem --system-site-packages: isolado
    pip("install", "--upgrade", "pip")
    if args.torch != "none":
        pip("install", "torch", "torchaudio", "torchvision", "--index-url", INDEX[args.torch])
    reqs = ["requirements.txt"]
    if args.torch != "none":
        reqs.append("requirements-engines.txt")  # openai-whisper exige o PyTorch
    if not args.no_quality:
        reqs.append("requirements-quality.txt")
    if not args.no_dev:
        reqs.append("requirements-dev.txt")
    pip("install", *[a for r in reqs for a in ("-r", os.path.join(ROOT, r))])
    if args.torch != "none" and not args.no_whisperx:
        pip("install", "--no-deps", "whisperx")
        pip("install", *WHISPERX_DEPS)

    print("\n== Verificação ==", flush=True)
    subprocess.call([PY, "-c", CHECK], cwd=ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
