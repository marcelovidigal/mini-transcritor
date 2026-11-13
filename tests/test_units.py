import numpy as np
import pytest

from transcritor import audio, media, srt
from transcritor.ollama_client import OllamaClient, build_prompt, build_translate_prompt


def test_format_timestamp():
    assert srt.format_timestamp(0) == "00:00:00,000"
    assert srt.format_timestamp(3661.5) == "01:01:01,500"
    assert srt.format_timestamp(59.9996) == "00:01:00,000"


def test_cues_for_chunk_proportional_and_contiguous():
    text = "Primeira frase bem comprida para ocupar espaço suficiente na legenda de teste. Segunda frase também longa o bastante para quebrar."
    cues = srt.cues_for_chunk(10.0, 30.0, text)
    assert len(cues) >= 2
    assert cues[0].start == 10.0 and cues[-1].end == 30.0
    for a, b in zip(cues, cues[1:]):
        assert a.end == pytest.approx(b.start)
    assert all(len(c.text) <= srt.MAX_CUE_CHARS for c in cues)


def test_clean_text_and_empty():
    assert srt.clean_text('Transcription: "olá   mundo"') == "olá mundo"
    assert srt.cues_for_chunk(0, 5, "   ") == []


def test_to_srt_format():
    out = srt.to_srt([srt.Cue(0, 1.5, "a"), srt.Cue(1.5, 3, "b")])
    assert out == "1\n00:00:00,000 --> 00:00:01,500\na\n\n2\n00:00:01,500 --> 00:00:03,000\nb\n"


def test_kind_from_extension():
    for ext in ("mkv", "avi", "mp4", "MOV"):
        assert media.kind_from_extension(f"x.{ext}") == "video"
    for ext in ("mp3", "flac", "wav", "m4a", "OGG"):
        assert media.kind_from_extension(f"x.{ext}") == "audio"
    assert media.kind_from_extension("x.txt") is None


def test_split_chunks_cuts_in_silence_and_drops_silent():
    sr = audio.SAMPLE_RATE
    rng = np.random.default_rng(0)
    sig = (rng.standard_normal(sr * 60) * 3000).astype(np.int16)
    sig[int(22 * sr) : int(23 * sr)] = 0  # pausa de 1 s em 22–23 s
    chunks = audio.split_chunks(sig, target=20, maximum=28)
    assert chunks[0].start == 0
    assert 22 <= chunks[0].end <= 23  # corte dentro da pausa
    assert all(c.end - c.start <= 28.01 for c in chunks)
    # contiguidade
    for a, b in zip(chunks, chunks[1:]):
        assert a.end == pytest.approx(b.start)
    assert audio.split_chunks(np.zeros(sr * 10, dtype=np.int16)) == []


def _speech(sr, parts):
    """parts: lista de (duração_s, falando). Ruído alto = 'fala'; zeros = pausa."""
    rng = np.random.default_rng(1)
    return np.concatenate(
        [(rng.standard_normal(int(d * sr)) * 3000).astype(np.int16) if on else np.zeros(int(d * sr), np.int16) for d, on in parts]
    )


def test_detect_speech_skips_silence_and_closes_short_pauses():
    sr = audio.SAMPLE_RATE
    sig = _speech(sr, [(5, False), (2, True), (0.2, False), (2, True), (3, False), (1.5, True), (4, False)])
    spans = audio.detect_speech(sig)
    assert len(spans) == 2  # a pausa de 0,2 s é fechada; a de 3 s separa
    assert 4.6 <= spans[0][0] / sr <= 4.8 and 9.4 <= spans[0][1] / sr <= 9.6
    assert spans[1][0] / sr >= 11.8 and spans[1][1] / sr <= 14.1
    assert audio.detect_speech(np.zeros(sr * 5, dtype=np.int16)) == []
    assert audio.detect_speech(np.zeros(10, dtype=np.int16)) == []


def test_split_by_speech_groups_to_target_and_cuts_at_pauses():
    sr = audio.SAMPLE_RATE
    parts = [(3, False)]
    for _ in range(8):  # 8 falas de 3 s separadas por pausas de 1 s
        parts += [(3, True), (1, False)]
    chunks = audio.split_by_speech(_speech(sr, parts), target=8, maximum=12)
    assert chunks[0].start >= 2.6  # silêncio inicial pulado (menos a folga de 0,3 s)
    assert all(c.end - c.start <= 12.01 for c in chunks)
    assert len(chunks) == 3  # 3+1+3+1+3 = ~11 s -> corta na pausa após atingir o alvo (8 s)
    for a, b in zip(chunks, chunks[1:]):  # os cortes caem nas pausas (1 s), nunca dentro de uma fala
        assert b.start - a.end >= 0.3  # pausa de 1 s menos 0,3 s de folga de cada lado
    assert audio.split_by_speech(np.zeros(sr * 10, dtype=np.int16)) == []


def test_split_by_speech_splits_long_continuous_speech():
    sr = audio.SAMPLE_RATE
    chunks = audio.split_by_speech(_speech(sr, [(70, True)]), target=20, maximum=28)
    assert len(chunks) >= 3 and all(c.end - c.start <= 28.01 for c in chunks)
    assert chunks[0].start < 0.2 and chunks[-1].end > 69.8


def test_probe_classifies(media_dir):
    assert media.probe(str(media_dir / "a.mp3")).kind == "audio"
    assert media.probe(str(media_dir / "a.flac")).kind == "audio"
    for ext in ("mp4", "mkv", "avi"):
        info = media.probe(str(media_dir / f"v.{ext}"))
        assert info.kind == "video" and info.duration > 45


def test_probe_errors(media_dir):
    for name in ("mudo.mp4", "texto.mp3", "naoexiste.mp3"):
        with pytest.raises(media.MediaError):
            media.probe(str(media_dir / name))


def test_prompts():
    assert "original language" in build_prompt()
    p = build_translate_prompt("pt-BR", "hello")
    assert "Brazilian Portuguese" in p and p.endswith("Text:\nhello")


def test_audio_models_filter(monkeypatch):
    c = OllamaClient("http://x")

    def fake(method, path, **kw):
        if path == "/api/tags":
            return {"models": [{"name": "a", "size": 2}, {"name": "b", "size": 1}, {"name": "c", "size": 3}]}
        caps = {"a": ["completion", "audio"], "b": ["completion", "vision", "audio"], "c": ["completion", "vision"]}
        return {"capabilities": caps[kw["json"]["model"]]}

    monkeypatch.setattr(c, "_request", fake)
    assert [m["name"] for m in c.audio_models()] == ["b", "a"]


def test_clean_env_strips_pyinstaller_bundle(monkeypatch, tmp_path):
    from transcritor import envutil

    bundle = str(tmp_path / "_internal")
    other = str(tmp_path / "ok")
    monkeypatch.setattr(envutil.sys, "_MEIPASS", bundle, raising=False)
    monkeypatch.setenv("PATH", bundle + envutil.os.pathsep + other)
    monkeypatch.setenv("LD_LIBRARY_PATH", bundle)
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/usr/lib")
    env = envutil.clean_env()
    assert env["PATH"] == other and env["LD_LIBRARY_PATH"] == "/usr/lib"
