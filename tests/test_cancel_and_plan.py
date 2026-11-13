"""Cancelamento imediato (chamadas ao Ollama cortadas, tela livre, fila) e plano de dispositivos por etapa do motor."""
import http.server
import threading
import time

import pytest

from transcritor import engines
from transcritor.jobs import Cancelled, Job, JobManager
from transcritor.ollama_client import OllamaClient, OllamaError

from .test_devices import ALL, ONLY_CPU, XPU, have  # noqa: F401  (fixture `have`)


# ---------- plano de dispositivos: hardware separado das capacidades do motor ----------

def stages(plan):
    return {p["stage"]: (p["device"], p["why"]) for p in plan}


def test_whisperx_plan_splits_recognition_from_alignment(have):
    have(XPU)
    plan = engines.engine_plan("whisperx", "auto")
    got = stages(plan)
    assert got["Reconhecimento (fala → texto)"][0] == "cpu" and "GPU Intel (XPU)" in got["Reconhecimento (fala → texto)"][1]
    assert got["Alinhamento por palavra e diarização"] == ("xpu", "")


def test_other_engines_plans(have, monkeypatch):
    have(XPU)
    assert [p["device"] for p in engines.engine_plan("faster-whisper")] == ["cpu"]
    assert [p["device"] for p in engines.engine_plan("openai-whisper")] == ["xpu"]
    from transcritor import whispercpp_manager as wm

    monkeypatch.setattr(wm, "has_nvidia", lambda: False)
    assert [p["device"] for p in engines.engine_plan("whisper-cpp")] == ["cpu"]
    have(ONLY_CPU)
    assert all(p["device"] == "cpu" and p["why"] == "" for p in engines.engine_plan("whisperx"))
    have(ALL)
    assert stages(engines.engine_plan("whisperx"))["Reconhecimento (fala → texto)"][0] == "cuda"
    assert engines.engine_plan("nao-existe") == []


def test_explicit_unavailable_device_is_reported_not_raised(have):
    have(ONLY_CPU)
    p = engines.engine_plan("openai-whisper", "xpu")[0]
    assert p["device"] is None and "xpu" in p["why"]


def test_devices_endpoint_returns_the_plan_for_the_engine(client_factory, have):
    have(XPU)
    c, _ = client_factory()
    body = c.get("/api/devices?engine=whisperx").get_json()
    assert [p["stage"] for p in body["plan"]] == ["Reconhecimento (fala → texto)", "Alinhamento por palavra e diarização"]
    assert c.get("/api/devices").get_json()["plan"] == []


# ---------- cancelamento ----------

class Slow(http.server.BaseHTTPRequestHandler):
    def do_POST(self):  # simula uma geração longa: só responde depois de 6 s
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        time.sleep(6)
        try:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"message": {"content": "tarde demais"}}')
        except OSError:
            pass

    def log_message(self, *a):
        pass


@pytest.fixture
def slow_server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Slow)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_abort_all_cuts_an_in_flight_ollama_call_immediately(slow_server):
    client = OllamaClient(slow_server, timeout=30)
    out = {}

    def call():
        t0 = time.perf_counter()
        try:
            client._chat({"model": "m", "messages": []})
        except BaseException as e:  # noqa: BLE001
            out["exc"] = e
        out["secs"] = time.perf_counter() - t0

    t = threading.Thread(target=call)
    t.start()
    time.sleep(0.5)
    assert client.abort_all() == 1
    t.join(5)
    assert isinstance(out["exc"], Cancelled) and out["secs"] < 3  # sem o corte seriam 6 s
    assert client.abort_all() == 0  # nada mais em andamento


def test_connection_refused_is_still_a_friendly_error():
    with pytest.raises(OllamaError, match="conectar"):
        OllamaClient("http://127.0.0.1:1", timeout=2)._request("GET", "/api/tags")


def wait_for(cond, secs=3.0):
    t0 = time.time()
    while time.time() - t0 < secs:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_cancelling_a_running_job_marks_it_cancelled_at_once_and_frees_the_ui():
    started, release = threading.Event(), threading.Event()
    log = {"cancel": 0, "finish": 0}

    def runner(job):
        started.set()
        release.wait(5)  # uma chamada longa que só volta depois
        job.check_cancel()

    mgr = JobManager(runner, on_finish=lambda j: log.__setitem__("finish", log["finish"] + 1),
                     on_cancel=lambda j: log.__setitem__("cancel", log["cancel"] + 1))
    a = mgr.create("a", "pt-BR", "m")
    assert started.wait(2)
    b = mgr.create("b", "pt-BR", "m")
    assert mgr.cancel(a.id) is True
    assert a.status == "cancelled" and a.is_finished and a.winding_down and log["cancel"] == 1  # imediato, sem esperar o worker
    d = mgr.describe(b)
    assert d["status"] == "queued" and "cancelamento anterior" in d["stage"]  # o próximo espera o worker sair
    release.set()
    assert wait_for(lambda: not a.winding_down)
    assert a.status == "cancelled" and log["finish"] >= 1
    assert wait_for(lambda: b.status == "cancelled" or b.status == "done" or b.status == "running")


def test_errors_caused_by_the_cut_count_as_cancellation_not_error():
    started, release = threading.Event(), threading.Event()

    def runner(job):
        started.set()
        release.wait(5)
        raise OllamaError("Erro de comunicação com o Ollama: conexão cortada")

    mgr = JobManager(runner, on_cancel=lambda j: None)
    a = mgr.create("a", "pt-BR", "m")
    assert started.wait(2)
    mgr.cancel(a.id)
    release.set()
    assert wait_for(lambda: not a.winding_down)
    assert a.status == "cancelled" and a.error is None


def test_cancelling_a_queued_job_still_works_and_unknown_job_is_false():
    gate = threading.Event()
    mgr = JobManager(lambda j: gate.wait(3))
    first = mgr.create("a", "pt-BR", "m")
    second = mgr.create("b", "pt-BR", "m")
    assert wait_for(lambda: first.status == "running")
    assert mgr.cancel(second.id) is True and second.status == "cancelled" and not second.winding_down
    assert mgr.cancel("nope") is False
    gate.set()


def test_interruptible_engine_call_returns_control_when_the_job_is_cancelled():
    job = Job(id="x", path="p", language="pt-BR", model="m")
    threading.Timer(0.3, job.cancel_event.set).start()
    t0 = time.perf_counter()
    with pytest.raises(Cancelled):
        engines._interruptible(job, time.sleep, 4)
    assert time.perf_counter() - t0 < 2
    job2 = Job(id="y", path="p", language="pt-BR", model="m")
    assert engines._interruptible(job2, lambda a, b=1: a + b, 1, b=2) == 3
    with pytest.raises(ValueError):
        engines._interruptible(job2, lambda: (_ for _ in ()).throw(ValueError("x")))
