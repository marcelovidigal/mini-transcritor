import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from transcritor import pipeline
from transcritor.audio import Chunk
from transcritor.config import Config
from transcritor.ollama_client import OllamaError, build_batch_translate_prompt, parse_batch
from transcritor.settings import Settings, SettingsError


class StubJob:
    model = "m:1b"
    language = "en"

    def __init__(self):
        self.warnings = []

    def check_cancel(self):
        pass


def chunks(n):
    return [Chunk(i * 10.0, i * 10.0 + 9, np.full(1600, i + 1, dtype=np.int16)) for i in range(n)]


def noop(*a, **k):
    pass


# ---------- transcrição em paralelo ----------

class SlowClient:
    """Trechos de índice par demoram mais: força a conclusão fora de ordem."""

    def __init__(self):
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def transcribe(self, model, wav):
        import wave, io

        with wave.open(io.BytesIO(wav)) as w:
            tag = int(np.frombuffer(w.readframes(1), dtype=np.int16)[0])
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.05 if tag % 2 else 0.15)
        with self.lock:
            self.active -= 1
        return f"texto {tag}"


def test_transcribe_all_keeps_order_and_runs_in_parallel():
    c = SlowClient()
    texts, failures, _ = pipeline._transcribe_all(StubJob(), c, chunks(6), Config(parallel=3), noop)
    assert texts == [f"texto {i}" for i in range(1, 7)]
    assert failures == 0 and 2 <= c.peak <= 3


def test_transcribe_all_parallel_1_is_sequential():
    c = SlowClient()
    pipeline._transcribe_all(StubJob(), c, chunks(4), Config(parallel=1), noop)
    assert c.peak == 1


def test_transcribe_all_aborts_when_first_attempts_fail():
    class Bad:
        def transcribe(self, model, wav):
            raise OllamaError("sem servidor")

    with pytest.raises(OllamaError):
        pipeline._transcribe_all(StubJob(), Bad(), chunks(8), Config(parallel=2), noop)


def test_transcribe_all_isolated_failure_is_a_warning():
    class Flaky(SlowClient):
        def transcribe(self, model, wav):
            t = super().transcribe(model, wav)
            if t == "texto 2":
                raise OllamaError("pontual")
            return t

    job = StubJob()
    texts, failures, last = pipeline._transcribe_all(job, Flaky(), chunks(5), Config(parallel=2), noop)
    assert failures == 1 and texts[1] == "" and texts[0] == "texto 1" and len(job.warnings) == 1 and last == "pontual"


# ---------- tradução em lote ----------

class BatchClient:
    def __init__(self, bad_batch=False, raise_batch=False):
        self.batch_sizes, self.singles = [], []
        self.bad_batch, self.raise_batch = bad_batch, raise_batch

    def translate_batch(self, model, texts, language):
        self.batch_sizes.append(len(texts))
        if self.raise_batch:
            raise OllamaError("formato")
        return texts[:-1] if self.bad_batch else [t.upper() for t in texts]

    def translate(self, model, text, language):
        self.singles.append(text)
        return text.upper()


def test_translate_all_batches_and_preserves_order():
    texts = [f"fala {i}" for i in range(20)]
    texts[3] = ""  # trecho sem texto fica de fora
    c = BatchClient()
    out = pipeline._translate_all(StubJob(), c, texts, Config(parallel=2, translate_batch=8), noop)
    assert sorted(c.batch_sizes, reverse=True) == [8, 8, 3] and not c.singles
    assert out[3] == "" and out[0] == "FALA 0" and out[19] == "FALA 19"


@pytest.mark.parametrize("kw", [{"bad_batch": True}, {"raise_batch": True}])
def test_translate_all_falls_back_to_single_calls(kw):
    texts = ["a b", "c d", "e f"]
    c = BatchClient(**kw)
    out = pipeline._translate_all(StubJob(), c, texts, Config(translate_batch=8), noop)
    assert out == ["A B", "C D", "E F"] and len(c.singles) == 3


def test_translate_all_keeps_original_when_translation_fails():
    class Fail(BatchClient):
        def translate(self, model, text, language):
            raise OllamaError("x")

    job = StubJob()
    out = pipeline._translate_all(job, Fail(raise_batch=True), ["um", "dois"], Config(translate_batch=8), noop)
    assert out == ["um", "dois"] and len(job.warnings) == 2


def test_translate_batch_size_1_uses_single_calls():
    c = BatchClient()
    pipeline._translate_all(StubJob(), c, ["a", "b", "c"], Config(translate_batch=1), noop)
    assert not c.batch_sizes and len(c.singles) == 3


def test_batches_respect_size_and_chars():
    texts = ["x" * 1000] * 5
    assert [len(b) for b in pipeline._batches(list(range(5)), texts, 8, max_chars=2500)] == [2, 2, 1]


# ---------- resposta do modelo ----------

def test_parse_batch_accepts_json_fences_and_wrapped_object():
    assert parse_batch('["a","b"]', 2) == ["a", "b"]
    assert parse_batch('```json\n["a","b"]\n```', 2) == ["a", "b"]
    assert parse_batch('{"translations": ["a","b"]}', 2) == ["a", "b"]


@pytest.mark.parametrize("raw", ["não é json", '["a"]', '["a", 2]', '{"x": 1}'])
def test_parse_batch_rejects_bad_shapes(raw):
    with pytest.raises(OllamaError):
        parse_batch(raw, 2)


def test_batch_prompt_mentions_count_and_language():
    p = build_batch_translate_prompt("pt-BR", ["hello", "world"])
    assert "2" in p and "Brazilian Portuguese" in p and '["hello", "world"]' in p


# ---------- configurações ----------

def test_settings_parallel_and_batch_validation(tmp_path):
    s = Settings(str(tmp_path / "s.json"))
    assert s["parallel"] == 2 and s["translate_batch"] == 8
    assert s.update({"parallel": 4, "translate_batch": 1})["parallel"] == 4
    for bad in ({"parallel": 0}, {"parallel": 5}, {"parallel": "x"}, {"translate_batch": 17}):
        with pytest.raises(SettingsError):
            s.update(bad)


# ---------- modelo carregado (keep_alive / pré-aquecimento) ----------

def test_chat_payloads_carry_keep_alive(monkeypatch):
    from transcritor.ollama_client import OllamaClient

    sent = []
    c = OllamaClient("http://x", keep_alive="45m")
    monkeypatch.setattr(c, "_request", lambda method, path, **kw: sent.append((path, kw["json"])) or {"message": {"content": "[\"a\"]"}})
    c.transcribe("m:1b", b"wav")
    c.translate("m:1b", "a", "en")
    c.translate_batch("m:1b", ["a"], "en")
    c.warm_up("m:1b")
    assert [p for p, _ in sent] == ["/api/chat"] * 3 + ["/api/generate"]
    assert all(body["keep_alive"] == "45m" for _, body in sent)
    assert "prompt" not in sent[-1][1]  # sem prompt: só carrega o modelo


def test_default_keep_alive_is_longer_than_ollama_default(monkeypatch):
    from transcritor.ollama_client import OllamaClient

    monkeypatch.delenv("MT_KEEP_ALIVE", raising=False)
    assert OllamaClient("http://x").keep_alive == "30m"
    monkeypatch.setenv("MT_KEEP_ALIVE", "2h")
    assert OllamaClient("http://x").keep_alive == "2h"


def test_warm_up_never_breaks_the_job():
    class Boom:
        def warm_up(self, model):
            raise OllamaError("sem memória")

    t = pipeline._start_warm_up(Boom(), "m:1b")
    t.join(2)
    assert not t.is_alive()
    assert pipeline._start_warm_up(object(), "m:1b") is None


def test_job_warms_up_the_model(client_factory, media_dir, tmp_path):
    import shutil
    from .test_api import start, wait_done

    src = tmp_path / "e.mp3"
    shutil.copy(media_dir / "a.mp3", src)
    c, fake = client_factory()
    s, _ = wait_done(c, start(c, src).get_json()["id"])
    assert s["status"] == "done" and fake.warmed == ["fake-audio:1b"]


# ---------- tradução só quando o texto NÃO está no idioma de saída ----------

PT = ["Hoje vamos conversar sobre como organizar o tempo durante a semana.",
      "O primeiro hábito é planejar o dia na noite anterior, com calma.",
      "Depois vamos falar sobre as pausas e o descanso que são necessários.",
      "Para terminar, lembre que ninguém muda a rotina de um dia para o outro."]
EN = ["The meeting will start at nine in the morning in the main office.",
      "We will review the budget for the next quarter and the new features."]


class PtJob(StubJob):
    language = "pt-BR"


def test_translate_skips_texts_already_in_the_target_language():
    c = BatchClient()
    out = pipeline._translate_all(PtJob(), c, PT, Config(translate_batch=8), noop)
    assert out == PT and not c.batch_sizes and not c.singles  # nenhuma chamada ao modelo


def test_translate_only_the_foreign_texts_in_a_mixed_file():
    c = BatchClient()
    texts = PT + EN
    out = pipeline._translate_all(PtJob(), c, texts, Config(translate_batch=8), noop)
    assert out[:4] == PT and out[4] == EN[0].upper() and out[5] == EN[1].upper()  # só os 2 em inglês foram traduzidos
    assert c.batch_sizes == [2]


def test_short_unidentifiable_text_follows_the_file_language_but_not_across_languages():
    c = BatchClient()
    out = pipeline._translate_all(PtJob(), c, PT + ["Sim."], Config(translate_batch=8), noop)
    assert out[-1] == "Sim." and not c.batch_sizes and not c.singles  # arquivo em português: "Sim." fica
    c2 = BatchClient()
    pipeline._translate_all(PtJob(), c2, EN + ["Yes."], Config(translate_batch=8), noop)
    assert c2.batch_sizes == [3]  # só 2 frases em inglês: sem maioria clara no idioma de saída, traduz tudo (curta incluída)


def test_langid_detects_main_languages_and_stays_silent_when_unsure():
    from transcritor import langid

    assert langid.detect(PT[0]) == "pt" and langid.detect(EN[0]) == "en"
    assert langid.detect("La reunión comenzará a las nueve de la mañana en la oficina.") == "es"
    assert langid.detect("La réunion commencera à neuf heures du matin dans le bureau.") == "fr"
    assert langid.detect("Die Besprechung beginnt um neun Uhr morgens im Büro.") == "de"
    assert langid.detect("これはテストです") == "ja" and langid.detect("안녕하세요 여러분") == "ko" and langid.detect("Привет всем друзьям") == "ru"
    assert langid.detect("Sim.") is None and langid.detect("") is None
    assert langid.same_language(PT[0], "pt-BR") is True and langid.same_language(PT[0], "en") is False
