"""NeMo-Speech.cpp (3 modelos) e Nemotron NVFP4 via servidor vLLM: gerenciador, parser, janelas, execução e cancelamento."""
import hashlib
import json
import os
import sys
import types

import numpy as np
import pytest

from transcritor import engines
from transcritor import nemocpp_manager as nm
from transcritor.engines import EngineError, NemoSpeechCppEngine, NemotronVllmEngine, Seg
from transcritor.jobs import Cancelled
from transcritor.settings import Settings, SettingsError

from .test_whispercpp import StubJob

REL = ["nemo-speech-0.3.0-windows-x86_64-cpu.zip", "nemo-speech-0.3.0-windows-x86_64-cuda.zip", "nemo-speech-0.3.0-windows-x86_64-vulkan.zip",
       "nemo-speech-0.3.0-windows-x86_64-vulkan.zip.sha256", "nemo-speech-0.3.0-linux-x86_64-vulkan.tar.gz", "nemo-speech-0.3.0-macos-aarch64-metal.tar.gz",
       "SHA256SUMS", "itn_configs.tar.bz2"]


def assets(names):
    return [{"name": n, "browser_download_url": "https://x.test/" + n, "size": 10} for n in names]


# ---------- gerenciador ----------

def test_pick_asset_by_backend_and_platform():
    assert nm.pick_asset(assets(REL), "vulkan", "windows", "x86_64")["name"] == "nemo-speech-0.3.0-windows-x86_64-vulkan.zip"  # não o .sha256
    assert nm.pick_asset(assets(REL), "cuda", "windows", "x86_64")["name"].endswith("cuda.zip")
    assert nm.pick_asset(assets(REL), "metal", "macos", "aarch64")["name"].endswith("macos-aarch64-metal.tar.gz")
    assert nm.pick_asset(assets(REL), "cuda", "linux", "x86_64") is None


def test_preferred_backends_follow_the_machine():
    assert nm.preferred_backends("windows", "x86_64", nvidia=True, vulkan=True) == ["cuda", "vulkan", "cpu"]
    assert nm.preferred_backends("windows", "x86_64", nvidia=False, vulkan=True) == ["vulkan", "cpu"]  # Intel Arc / AMD
    assert nm.preferred_backends("windows", "x86_64", nvidia=False, vulkan=False) == ["vulkan", "cpu"]  # sem informação: tenta Vulkan
    assert nm.preferred_backends("macos", "aarch64", nvidia=False, vulkan=False) == ["metal", "cpu"]
    assert nm.preferred_backends("macos", "x86_64") == ["cpu"]


def test_find_release_asset_skips_prereleases_and_falls_back_between_backends(monkeypatch):
    monkeypatch.setattr(nm, "machine", lambda: ("windows", "x86_64"))  # os pacotes do teste são do Windows, qualquer que seja o SO do CI
    class R:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    rels = [{"tag_name": "nightly", "prerelease": True, "assets": assets(["nemo-speech-nightly-windows-x86_64-vulkan.zip"])},
            {"tag_name": "v0.3.0", "prerelease": False, "assets": assets(["nemo-speech-0.3.0-windows-x86_64-cpu.zip"])}]
    http = types.SimpleNamespace(get=lambda *a, **k: R(rels))
    got = nm.find_release_asset(["vulkan", "cpu"], http)
    assert got["tag"] == "v0.3.0" and got["backend"] == "cpu"  # nightly ignorado; sem Vulkan nesse release, cai para a CPU


# ---------- parser da saída JSON ----------

def words(spec):
    return [{"word": w, "start": s, "end": e, "confidence": 1} for w, s, e in spec]


def test_words_to_segments_breaks_on_pause_sentence_end_and_length():
    w = words([("Olá", 0.0, 0.4), ("mundo.", 0.5, 1.0), ("Tudo", 1.2, 1.5), ("bem", 1.6, 2.0), ("hoje", 2.1, 2.9), ("sim.", 3.0, 3.3),
               ("Depois", 6.0, 6.5), ("de", 6.6, 6.7)])
    segs = engines.words_to_segments(w, offset=100.0)
    assert [s.text for s in segs] == ["Olá mundo. Tudo bem hoje sim.", "Depois de"]  # "mundo." só 1 s: não quebra; pausa de 2,7 s quebra
    assert segs[0].start == 100.0 and segs[0].end == pytest.approx(103.3) and segs[1].start == pytest.approx(106.0)
    long = words([(f"p{i}", i * 0.5, i * 0.5 + 0.4) for i in range(60)])
    assert all(s.end - s.start <= 12.5 for s in engines.words_to_segments(long)) and len(engines.words_to_segments(long)) >= 3


def test_parse_nemo_json_shapes_and_language():
    segs, lang = engines.parse_nemo_json({"words": words([("Oi", 1.0, 1.3)]), "languages": ["pt-PT", "pt-BR"]}, 10.0)
    assert segs == [Seg(11.0, 11.3, "Oi")] and lang == "pt"
    segs, lang = engines.parse_nemo_json({"segments": [{"start": 0, "end": 2, "text": " ola "}, {"start": 3, "end": 4, "text": ""}]}, 5.0)
    assert segs == [Seg(5.0, 7.0, "ola")] and lang is None
    segs, _ = engines.parse_nemo_json({"text": "só texto", "duration": 4.0})
    assert segs == [Seg(0.0, 4.0, "só texto")]
    assert engines.parse_nemo_json({"text": "", "words": []}) == ([], None)


def test_window_chunks_respect_the_model_limit():
    ch = [types.SimpleNamespace(start=s, end=s + 10.0) for s in range(0, 100, 10)]
    assert engines.window_chunks(ch, 1000.0) == [(0, 100.0)]
    assert engines.window_chunks(ch, 45.0) == [(0, 40.0), (40, 80.0), (80, 100.0)]
    assert engines.window_chunks([], 10) == []


def test_format_elapsed():
    assert engines.format_elapsed(75) == "1:15" and engines.format_elapsed(3725) == "1:02:05"


# ---------- catálogo e modelos ----------

def test_catalog_lists_the_three_cpp_models_and_the_nvfp4_server():
    cpp = {c["name"] for c in engines.load_catalog()["nemo-speech-cpp"]}
    assert cpp == set(engines.NEMO_MODELS) and len(cpp) == 3
    assert engines.NEMO_MODELS["nemotron-3.5-transcribe-30b-a4b-q2_0_m"]["files"][0]["size"] < engines.NEMO_MODELS["nemotron-3.5-transcribe-30b-a4b-q2_0_l"]["files"][0]["size"]
    assert [c["name"] for c in engines.load_catalog()["nemotron-vllm"]] == [NemotronVllmEngine.MODEL]
    for s in engines.NEMO_MODELS.values():  # todo arquivo tem tamanho, SHA-256 e URL com revisão fixa (nunca "main")
        for f in s["files"]:
            assert len(f["sha256"]) == 64 and f["size"] > 1e8 and "/resolve/main/" not in f["url"]
    salm = engines.NEMO_MODELS["nemotron-3.5-transcribe-30b-a4b-q2_0_m"]["files"]
    assert len(salm) == 2 and "Encoder" in salm[1]["name"]  # par decodificador + codificador


def test_nemo_models_are_not_installed_until_files_have_the_exact_size(tmp_path, monkeypatch):
    eng = NemoSpeechCppEngine()
    monkeypatch.setattr(NemoSpeechCppEngine, "models_dir", staticmethod(lambda: str(tmp_path)))
    name = "nemotron-3.5-asr-streaming-0.6b"
    assert not eng.is_installed(name)
    f = tmp_path / name / engines.NEMO_MODELS[name]["files"][0]["name"]
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x" * 10)
    assert not eng.is_installed(name)  # truncado
    assert eng.model_file(name) == str(f)


def test_download_verifies_size_and_sha_and_resumes(tmp_path, monkeypatch):
    data = b"abc" * 100_000
    sha = hashlib.sha256(data).hexdigest()
    calls = []

    class Resp:
        def __init__(self, status, body):
            self.status_code, self.body, self.headers = status, body, {}

        def raise_for_status(self):
            pass

        def iter_content(self, n):
            yield self.body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def fake_get(url, **kw):
        rng = kw.get("headers", {}).get("Range")
        calls.append(rng)
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        return Resp(206 if rng else 200, data[start:])

    import requests
    monkeypatch.setattr(requests, "get", fake_get)
    target = tmp_path / "m.gguf"
    (tmp_path / "m.gguf.part").write_bytes(data[:1000])  # download interrompido antes
    got = []
    engines._download("https://x.test/m", str(target), len(data), sha, lambda d, t: got.append((d, t)))
    assert target.read_bytes() == data and calls == ["bytes=1000-"] and got[-1] == (len(data), len(data)) and not (tmp_path / "m.gguf.part").exists()
    with pytest.raises(EngineError, match="corrompido"):
        engines._download("https://x.test/m", str(tmp_path / "n.gguf"), len(data), "0" * 64)
    assert not (tmp_path / "n.gguf").exists()
    with pytest.raises(EngineError, match="incompleto"):
        engines._download("https://x.test/m", str(tmp_path / "o.gguf"), len(data) + 5, sha)


# ---------- execução (subprocesso) ----------

def test_run_returns_json_and_reports_errors():
    eng = NemoSpeechCppEngine()
    job = StubJob()
    ticks = []
    out = eng._run([sys.executable, "-c", "import json;print(json.dumps({'text':'oi','words':[]}))"], job, ticks.append)
    assert out["text"] == "oi"
    with pytest.raises(EngineError, match="boom"):
        eng._run([sys.executable, "-c", "import sys,json;print(json.dumps({'error':{'message':'boom'}}));sys.exit(1)"], job, ticks.append)


def test_run_kills_the_process_on_cancel():
    eng = NemoSpeechCppEngine()
    job = StubJob(cancel_after=2)
    import time
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        eng._run([sys.executable, "-c", "import time;time.sleep(60)"], job, lambda el: None)
    assert time.monotonic() - t0 < 10  # o subprocesso foi morto, não esperado


def test_transcribe_windows_offsets_and_language(tmp_path, monkeypatch):
    eng = NemoSpeechCppEngine()
    name = "nemotron-3.5-asr-streaming-0.6b"
    monkeypatch.setattr(eng, "find_binary", lambda: "nemo-speech")
    monkeypatch.setattr(eng, "is_installed", lambda m: True)
    seen = []

    def fake_run(cmd, job, tick):
        seen.append(cmd)
        return {"words": words([("olá", 1.0, 1.4), ("pessoal", 1.5, 2.0)]), "languages": ["pt-BR"]}

    monkeypatch.setattr(eng, "_run", fake_run)
    monkeypatch.setitem(engines.NEMO_MODELS[name], "window", 25.0)  # força 2 janelas
    chunks = [types.SimpleNamespace(start=0.0, end=10.0, samples=None), types.SimpleNamespace(start=12.0, end=22.0, samples=None),
              types.SimpleNamespace(start=30.0, end=40.0, samples=None)]
    samples = np.zeros(16000 * 40, dtype=np.int16)
    job = StubJob()
    segs, lang = eng.transcribe(name, samples, chunks, job, lambda p, s=None: None, context="termos")
    assert lang == "pt" and [round(s.start) for s in segs] == [1, 31]  # 2ª janela começa em 30 s
    assert len(seen) == 2 and "--language" in seen[0] and "auto" in seen[0] and "--json" in seen[0]
    assert any("não usa o contexto" in w for w in job.warnings)  # o 0,6 B ignora o glossário e avisa


def test_salm_runs_one_batch_over_the_vad_chunks_and_keeps_vad_timing(monkeypatch):
    eng = NemoSpeechCppEngine()
    name = "nemotron-3.5-transcribe-30b-a4b-q2_0_m"
    monkeypatch.setattr(eng, "find_binary", lambda: "nemo-speech")
    monkeypatch.setattr(eng, "is_installed", lambda m: True)
    seen = []

    def fake_run(cmd, job, tick, prog=None, want_json=True):
        seen.append(cmd)
        assert not want_json and len(os.listdir(cmd[2])) == 3  # uma pasta com 1 wav por trecho
        out = cmd[cmd.index("--output-dir") + 1]
        os.makedirs(out)
        texts = ["Bom dia a todos, vamos começar a reunião de hoje.", "", "Obrigado pela atenção de vocês, até a próxima semana."]
        for i, tx in enumerate(texts):
            if i != 1:  # o 2º trecho não gerou arquivo: vira aviso, não erro
                with open(os.path.join(out, f"c{i:06d}.json"), "w", encoding="utf-8") as f:
                    json.dump({"text": tx, "words": [], "duration": 5.0}, f)
        prog and prog.update(k=3, n=3)
        tick(1.0)

    monkeypatch.setattr(eng, "_run", fake_run)
    chunks = [types.SimpleNamespace(start=s, end=s + 5.0, samples=np.zeros(16000 * 5, dtype=np.int16)) for s in (0.0, 10.0, 20.0)]
    job = StubJob()
    stages = []
    segs, lang = eng.transcribe(name, np.zeros(16000 * 30, dtype=np.int16), chunks, job, lambda p, s=None: stages.append(s), context="Dra. Helena Duarte")
    cmd = seen[0]
    assert "--instruction" in cmd and "Dra. Helena Duarte" in cmd[cmd.index("--instruction") + 1] and "--language" not in cmd
    assert "--concurrency" in cmd and "--format" in cmd
    assert [(s.start, s.end) for s in segs] == [(0.0, 5.0), (20.0, 25.0)]  # tempo do VAD
    assert lang == "pt" and any("Trecho 2" in w for w in job.warnings) and "trecho 3 de 3" in stages[-1]


def test_chunk_segments_distrust_collapsed_word_times():
    ch = types.SimpleNamespace(start=100.0, end=112.0)
    good = {"text": "Olá a todos. Tudo bem hoje?", "words": words([("Olá", 0.2, 0.5), ("a", 0.5, 0.6), ("todos.", 0.6, 1.1), ("Tudo", 3.0, 3.3), ("bem", 3.3, 3.6), ("hoje?", 3.6, 4.2)])}
    segs = engines.chunk_segments(good, ch)
    assert [s.text for s in segs] == ["Olá a todos.", "Tudo bem hoje?"] and segs[0].start == pytest.approx(100.2)  # pausa de 1,9 s separa
    packed = {"text": "mil setecentos e sessenta e sete foram os trilhos", "words": words(
        [("mil", 0.0, 0.05), ("setecentos", 0.05, 0.06), ("sessenta", 19.0, 19.05), ("e", 19.05, 19.1), ("sete", 19.1, 19.2)])}  # salto de 19 s
    assert engines.chunk_segments(packed, ch) == [Seg(100.0, 112.0, packed["text"])]  # volta ao intervalo do trecho
    squeezed = {"text": " ".join(f"p{i}" for i in range(12)), "words": words([(f"p{i}", 0.01 * i, 0.01 * i + 0.01) for i in range(12)])}
    assert engines.chunk_segments(squeezed, ch) == [Seg(100.0, 112.0, squeezed["text"])]  # tudo espremido no início
    crowded = {"text": "Frase um bem curta. Esta outra frase é bem mais longa e foi amontoada no fim do trecho pelo alinhador.", "words": words(
        [("Frase", 0.2, 0.5), ("um", 0.5, 0.7), ("bem", 0.7, 1.0), ("curta.", 1.0, 1.6)] + [(f"w{i}", 9.5 + 0.1 * i, 9.6 + 0.1 * i) for i in range(16)])}
    assert engines.chunk_segments(crowded, ch) == [Seg(100.0, 112.0, crowded["text"])]  # a 2ª frase passa de 20 caracteres/s: fica o trecho inteiro
    assert engines.chunk_segments({"text": "  "}, ch) == [] and engines.chunk_segments({"text": "só texto"}, ch) == [Seg(100.0, 112.0, "só texto")]


def test_run_reads_progress_lines_from_stderr():
    eng = NemoSpeechCppEngine()
    prog = {"k": 0, "n": 0}
    code = "import sys; sys.stderr.write('[1/3] a.wav -> a.json' + chr(10) + '[3/3] c.wav -> c.json' + chr(10))"
    eng._run([sys.executable, "-c", code], StubJob(), lambda el: None, prog, want_json=False)
    assert prog == {"k": 3, "n": 3}


def test_device_flag_mapping(monkeypatch):
    monkeypatch.delenv("MT_DEVICE", raising=False)
    f = NemoSpeechCppEngine._device_flag
    assert f("auto") == [] and f("cpu") == ["--device", "cpu"] and f("cuda") == ["--device", "cuda:0"] and f("xpu") == ["--device", "vulkan:0"]
    monkeypatch.setenv("MT_DEVICE", "cpu")
    assert f("auto") == ["--device", "cpu"]


def test_plan_names_the_backend_not_pytorch(monkeypatch):
    monkeypatch.delenv("MT_DEVICE", raising=False)
    monkeypatch.setattr(nm, "installed_backend", lambda: "vulkan")
    p = engines.engine_plan("nemo-speech-cpp", "auto")
    assert p[0]["device"] == "gpu" and "Vulkan" in p[0]["label"]
    assert engines.engine_plan("nemo-speech-cpp", "cpu")[0]["device"] == "cpu"


# ---------- servidor vLLM (NVFP4) ----------

def test_vllm_engine_availability_and_transcription(monkeypatch):
    import requests

    eng = NemotronVllmEngine()
    posted = []

    class R:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    def fake_get(url, **kw):
        if url.endswith("/models"):
            return R({"data": [{"id": NemotronVllmEngine.MODEL}]})
        raise AssertionError(url)

    def fake_post(url, json=None, **kw):
        posted.append((url, json))
        return R({"choices": [{"message": {"content": " Bom dia a todos, vamos começar a reunião de hoje. "}}]})

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)
    assert eng.available() and eng.is_installed(NemotronVllmEngine.MODEL) and not eng.is_installed("outro")
    chunks = [types.SimpleNamespace(start=5.0, end=9.0, samples=np.zeros(16000 * 4, dtype=np.int16))]
    segs, lang = eng.transcribe(NemotronVllmEngine.MODEL, None, chunks, StubJob(), lambda p, s=None: None)
    assert segs == [Seg(5.0, 9.0, "Bom dia a todos, vamos começar a reunião de hoje.")] and lang == "pt"
    url, body = posted[0]
    assert url.endswith("/chat/completions") and body["temperature"] == 0 and body["messages"][0]["content"][1]["type"] == "audio_url"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_vllm_engine_unavailable_message(monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("recusada")

    monkeypatch.setattr(requests, "get", boom)
    eng = NemotronVllmEngine()
    assert not eng.available() and "Blackwell" in eng.not_installed_message() and eng.url().startswith("http://127.0.0.1:8000")
    eng.configured_url = "http://servidor:9000/v1/"
    assert eng.url() == "http://servidor:9000/v1"
    eng.configured_url = ""
    assert "Blackwell" in (engines.check_request("nemotron-vllm", NemotronVllmEngine.MODEL, False, "") or "")


def test_settings_validate_new_paths(tmp_path):
    s = Settings(str(tmp_path / "s.json"))
    assert s["nemocpp_path"] == "" and s["nemotron_vllm_url"] == ""
    with pytest.raises(SettingsError):
        s.update({"nemocpp_path": str(tmp_path / "nada")})
    with pytest.raises(SettingsError):
        s.update({"nemotron_vllm_url": "servidor:8000"})
    assert s.update({"nemotron_vllm_url": "http://10.0.0.5:8000/v1/"})["nemotron_vllm_url"] == "http://10.0.0.5:8000/v1"


def test_engines_are_listed_for_the_screen_and_vllm_hides_until_a_server_answers(monkeypatch):
    import requests

    monkeypatch.setattr(NemotronVllmEngine, "served", lambda self: [])
    ids = {e["id"] for e in engines.all_status()}
    assert "nemo-speech-cpp" in ids and "nemotron-vllm" not in ids  # sem servidor: escondido
    assert engines.get("nemotron-vllm") is not None  # mas continua cadastrado (API e variável NEMOTRON_VLLM_URL)
    monkeypatch.setattr(NemotronVllmEngine, "served", lambda self: [NemotronVllmEngine.MODEL])
    assert "nemotron-vllm" in {e["id"] for e in engines.all_status()}


# ---------- ruído de avisos, teto de geração e contexto do Ollama ----------

def test_review_failures_become_a_single_counted_warning(caplog):
    from transcritor import quality_engine as qe

    job = StubJob()
    job.warnings = []
    for i in range(1, 6):
        qe._review_failed(job, i, "A revisão não devolveu JSON válido.", "mantido o texto atual")
    assert job.warnings == ["5 revisão(ões) de fragmento sem resposta válida do revisor; mantido o texto atual."]  # uma linha, não cinco
    assert "fragmento 3" in caplog.text  # o detalhe por fragmento vai para o log


def test_generation_is_capped_by_audio_duration():
    from transcritor import audio
    from transcritor import ollama_client as oc

    short = audio.to_wav_bytes(np.zeros(16000 * 2, dtype=np.int16))
    long = audio.to_wav_bytes(np.zeros(16000 * 28, dtype=np.int16))
    assert oc.token_cap(short) == 128 and oc.token_cap(long) == 28 * oc.TOKENS_PER_SECOND + 96 and oc.token_cap(b"") == 128
    assert oc.token_cap(audio.to_wav_bytes(np.zeros(16000 * 600, dtype=np.int16))) == 1024
    assert oc.text_cap("a" * 100, "b" * 100) == 456 and oc.text_cap("x" * 10_000) == 4096


def test_every_chat_call_carries_the_same_explicit_context_and_a_token_cap(monkeypatch):
    from transcritor import audio
    from transcritor import ollama_client as oc

    c = oc.OllamaClient("http://x")
    sent = []
    monkeypatch.setattr(c, "_request", lambda method, path, **kw: sent.append((path, kw["json"])) or {"message": {"content": "[]"}, "response": ""})
    wav = audio.to_wav_bytes(np.zeros(16000 * 5, dtype=np.int16))
    c.transcribe("m", wav)
    c.transcribe_variant("m", wav)
    for fn in (lambda: c.review_self("m", wav, "texto", 5), lambda: c.translate("m", "olá", "en"), lambda: c.translate_batch("m", ["olá"], "en"),
               lambda: c.polish("m", ["olá"], "pt-BR"), lambda: c.triage("m", [], "")):
        try:
            fn()
        except oc.OllamaError:
            pass  # a resposta falsa não é o JSON esperado; só interessa o que foi enviado
    c.warm_up("m")
    assert len(sent) >= 8
    for path, body in sent:
        opts = body["options"]
        assert opts["num_ctx"] == oc.NUM_CTX == 32768, path  # o mesmo valor em todas as chamadas (nenhum recarrega o modelo)
        assert "num_predict" in opts or path == "/api/generate", (path, body.get("model"))
    monkeypatch.setenv("MT_NUM_CTX", "65536")
    assert oc.OllamaClient("http://x").num_ctx == 65536


def test_review_schemas_bound_every_free_text_field():
    from transcritor import ollama_client as oc

    for s in (oc.S_ANALYSIS, oc.S_CRITIQUE, oc.S_TEXT):
        assert s["maxLength"] > 0  # gramática do Ollama: o JSON sempre fecha dentro do limite de tokens
    assert oc.S_TERMS["maxItems"] and oc.S_FIXES["maxItems"] and oc.S_FIXES["items"]["properties"]["from"]["maxLength"]


# ---------- log em arquivo ----------

def test_job_lifecycle_and_warnings_go_to_the_log_file(tmp_path):
    import logging

    from transcritor import applog, pipeline
    from transcritor.jobs import Job

    path = str(tmp_path / "logs" / "t.log")
    assert applog.setup(path) == path and applog.setup(path) == path  # idempotente
    assert sum(1 for h in logging.getLogger("transcritor").handlers if getattr(h, "_mt_file", None) == path) == 1
    job = Job(id="abc123", path=str(tmp_path / "filme secreto.mkv"), language="pt-BR", model="gemma4:e4b", engine="quality-1")
    job.warnings = ["3 revisão(ões) de fragmento sem resposta válida do revisor; mantido o texto atual."]

    def boom(j, c, cfg):
        raise EngineError("falha de teste")

    orig = pipeline._run_job
    pipeline._run_job = boom
    try:
        with pytest.raises(EngineError):
            pipeline.run_job(job, None, None)
    finally:
        pipeline._run_job = orig
        for h in logging.getLogger("transcritor").handlers:
            h.flush()
    text = open(path, encoding="utf-8").read()
    applog.close()
    assert "[abc123] início: arquivo=filme secreto.mkv motor=quality-1 modelo=gemma4:e4b" in text
    assert "[abc123] erro: falha de teste" in text and "[abc123] aviso: 3 revisão(ões)" in text
    assert str(tmp_path) not in text.split("início:")[1].split("\n")[0]  # só o nome do arquivo, não a pasta
