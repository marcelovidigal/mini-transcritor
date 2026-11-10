import subprocess
import sys
import os
import tempfile

os.environ["MINITRANSCRITOR_FFMPEG_DIR"] = tempfile.mkdtemp(prefix="mt_test_ff_")  # nunca baixa/escreve no projeto
os.environ["MINITRANSCRITOR_HOME"] = tempfile.mkdtemp(prefix="mt_test_home_")  # nunca toca nas settings reais

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app  # noqa: E402
from transcritor.config import Config  # noqa: E402


class FakeClient:
    """Dublê do OllamaClient."""

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def audio_models(self):
        return [{"name": "fake-audio:1b", "size": 1_000_000_000}] + getattr(self, "extra_models", [])

    def transcribe(self, model, wav_bytes):
        from transcritor.ollama_client import OllamaError

        if self.fail:
            raise OllamaError("falha simulada")
        self.calls.append((model, len(wav_bytes)))
        return f"Texto do trecho número {len(self.calls)}. Segunda frase do trecho."

    def pull(self, model, progress=None):
        self.pulled = getattr(self, "pulled", []) + [model]
        progress(5, 10, "pulling")
        progress(10, 10, "success")
        self.extra_models = getattr(self, "extra_models", []) + [{"name": model, "size": 1}]

    def translate(self, model, text, language):
        self.translations = getattr(self, "translations", []) + [language]
        return text


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("media")
    tone = ["-f", "lavfi", "-i", "sine=frequency=440:duration=50"]  # 50 s => vários trechos
    _ffmpeg(*tone, str(d / "a.mp3"))
    _ffmpeg(*tone, str(d / "a.flac"))
    for ext in ("mp4", "mkv", "avi"):
        _ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=50", *tone, "-shortest", str(d / f"v.{ext}"))
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=3", str(d / "mudo.mp4"))  # sem áudio
    _ffmpeg("-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "5", str(d / "silencio.wav"))
    (d / "texto.mp3").write_text("não sou mp3")
    return d


@pytest.fixture
def client_factory():
    def make(fail=False):
        fake = FakeClient(fail)
        app = create_app(Config(), fake)
        app.testing = True
        return app.test_client(), fake

    return make
