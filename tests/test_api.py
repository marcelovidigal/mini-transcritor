import os
import re
import shutil
import time


def wait_done(client, job_id, timeout=60):
    seen = []
    end = time.time() + timeout
    while time.time() < end:
        s = client.get(f"/api/jobs/{job_id}").get_json()
        seen.append(s["progress"])
        if s["status"] not in ("queued", "running"):
            return s, seen
        time.sleep(0.05)
    raise AssertionError("timeout")


def start(client, path, **over):
    body = {"path": str(path), "language": "pt-BR", "model": "fake-audio:1b", **over}
    return client.post("/api/jobs", json=body)


def test_index_and_lists(client_factory):
    c, _ = client_factory()
    assert b"Mini Transcritor" in c.get("/").data
    assert c.get("/api/models").get_json()["models"][0]["name"] == "fake-audio:1b"
    assert len(c.get("/api/languages").get_json()) > 3


def test_audio_job_generates_valid_srt(client_factory, media_dir, tmp_path):
    src = tmp_path / "entrada.mp3"
    shutil.copy(media_dir / "a.mp3", src)
    c, fake = client_factory()
    r = start(c, src)
    assert r.status_code == 202
    s, seen = wait_done(c, r.get_json()["id"])
    assert s["status"] == "done", s
    assert seen == sorted(seen) and s["progress"] == 100
    assert len(fake.calls) >= 2 and sum(fake.batches) == len(fake.calls)  # tradução em lote: uma chamada cobre as falas
    out = tmp_path / "entrada.pt-BR.srt"
    assert s["output_path"] == str(out) and out.exists()
    text = out.read_text(encoding="utf-8")
    blocks = text.strip().split("\n\n")
    for i, b in enumerate(blocks, 1):
        lines = b.split("\n")
        assert lines[0] == str(i)
        assert re.fullmatch(r"\d\d:\d\d:\d\d,\d{3} --> \d\d:\d\d:\d\d,\d{3}", lines[1])
        assert lines[2].strip()
    dl = c.get(f"/api/jobs/{s['id']}/download")
    assert dl.status_code == 200 and dl.data.decode("utf-8") == text


def test_video_formats(client_factory, media_dir, tmp_path):
    for ext in ("mp4", "mkv", "avi"):
        src = tmp_path / f"video.{ext}"
        shutil.copy(media_dir / f"v.{ext}", src)
        c, fake = client_factory()
        s, _ = wait_done(c, start(c, src).get_json()["id"])
        assert s["status"] == "done", (ext, s)
        assert os.path.exists(tmp_path / "video.pt-BR.srt")
        assert fake.calls


def test_validation_errors(client_factory, media_dir):
    c, _ = client_factory()
    assert start(c, "").status_code == 400
    assert start(c, media_dir / "nao.mp3").status_code == 400
    assert start(c, media_dir / "a.mp3", language="xx").status_code == 400
    assert start(c, media_dir / "a.mp3", model="outro").status_code == 400
    assert start(c, media_dir / "a.mp3", model="").status_code == 400


def test_job_errors(client_factory, media_dir):
    c, _ = client_factory()
    s, _ = wait_done(c, start(c, media_dir / "mudo.mp4").get_json()["id"])
    assert s["status"] == "error" and "áudio" in s["error"]
    s, _ = wait_done(c, start(c, media_dir / "silencio.wav").get_json()["id"])
    assert s["status"] == "error" and "silencioso" in s["error"]
    c, _ = client_factory(fail=True)
    s, _ = wait_done(c, start(c, media_dir / "a.mp3").get_json()["id"])
    assert s["status"] == "error" and "falha simulada" in s["error"]


def test_unknown_job(client_factory):
    c, _ = client_factory()
    assert c.get("/api/jobs/zzz").status_code == 404
    assert c.get("/api/jobs/zzz/download").status_code == 404


def test_cancel(client_factory, media_dir):
    import threading

    c, fake = client_factory()
    orig = fake.transcribe
    gate = threading.Event()

    def slow(*a):
        gate.wait(5)
        return orig(*a)

    fake.transcribe = slow
    jid = start(c, media_dir / "a.mp3").get_json()["id"]
    time.sleep(0.5)
    assert c.post(f"/api/jobs/{jid}/cancel").status_code == 200
    gate.set()
    s, _ = wait_done(c, jid)
    assert s["status"] == "cancelled"


def test_browse(client_factory, media_dir):
    c, _ = client_factory()
    d = c.get(f"/api/browse?path={media_dir}").get_json()
    names = {f["name"] for f in d["files"]}
    assert {"a.mp3", "v.mkv"} <= names and "texto.mp3" in names
    assert c.get("/api/browse?path=/nao/existe/xyz").status_code == 400
    assert c.get("/api/browse").get_json()["ok"]
