import io
import os
import threading
import time

import pytest

from app import create_app
from transcritor.config import Config
from transcritor.webserver import ConfigError, ServerConfig, sweep

from .conftest import FakeClient

H = {"X-Requested-With": "fetch"}
ALICE, BOB = "alice@exemplo.com", "bob@exemplo.com"


def cfg_for(tmp_path, **over):
    base = dict(secret_key="x" * 32, google_client_id="id", google_client_secret="sec",
                allowed_emails={ALICE, BOB}, data_dir=str(tmp_path / "data"), max_upload_mb=5, max_duration_min=180)
    base.update(over)
    return ServerConfig(**base)


class Server:
    def __init__(self, tmp_path, fake=None, **over):
        self.cfg = cfg_for(tmp_path, **over)
        self.fake = fake or FakeClient()
        self.app = create_app(Config(), self.fake, server=self.cfg)
        self.app.testing = True
        self.manager = self.app.extensions["job_manager"]

    def client(self, email=ALICE):
        c = self.app.test_client()
        if email:
            with c.session_transaction() as s:
                s["user"] = {"email": email, "name": email}
        return c

    @property
    def upload(self):
        return os.listdir(self.cfg.upload_dir)

    @property
    def download(self):
        return os.listdir(self.cfg.download_dir)


@pytest.fixture
def srv(tmp_path):
    return Server(tmp_path)


def upload(c, path, name="aula.mp3", language="pt-BR", model="fake-audio:1b"):
    with open(path, "rb") as f:
        return c.post("/api/jobs", headers=H, content_type="multipart/form-data",
                      data={"file": (io.BytesIO(f.read()), name), "language": language, "model": model})


def wait(c, job_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        s = c.get(f"/api/jobs/{job_id}").get_json()
        if s["status"] not in ("queued", "running"):
            return s
        time.sleep(0.05)
    raise AssertionError("timeout")


# ---------- configuração ----------
def test_config_from_env_fail_closed():
    with pytest.raises(ConfigError) as e:
        ServerConfig.from_env({})
    msg = str(e.value)
    assert "GOOGLE_CLIENT_ID" in msg and "SECRET_KEY" in msg and "ALLOWED_EMAILS" in msg
    ok = ServerConfig.from_env({"GOOGLE_CLIENT_ID": "a", "GOOGLE_CLIENT_SECRET": "b", "SECRET_KEY": "k" * 20,
                                "ALLOWED_EMAILS": " A@x.com , b@x.com", "ALLOWED_DOMAINS": "Empresa.com",
                                "BASE_URL": "https://t.exemplo.com/"})
    assert ok.allowed_emails == {"a@x.com", "b@x.com"} and ok.cookie_secure and ok.base_url == "https://t.exemplo.com"
    assert ok.is_allowed("QUALQUER@empresa.com") and ok.is_allowed("a@x.com") and not ok.is_allowed("c@x.com")
    assert not ok.is_allowed("")


# ---------- acesso ----------
def test_requires_login(srv):
    c = srv.client(None)
    r = c.get("/")
    assert r.status_code == 302 and "/login" in r.headers["Location"]
    assert c.get("/api/models").status_code == 401
    assert c.post("/api/jobs").status_code == 401
    assert c.get("/healthz").status_code == 200
    assert b"Entrar com Google" in c.get("/login").data


def test_local_only_routes_do_not_exist(srv):
    c = srv.client()
    assert c.get("/api/browse").status_code == 404
    assert c.get("/api/settings").status_code == 404
    assert c.post("/api/settings", headers=H, json={}).status_code == 404
    assert c.post("/api/setup/install-ollama", headers=H).status_code == 404


def test_csrf_header_required_for_writes(srv):
    c = srv.client()
    assert c.post("/api/jobs", data={}).status_code == 403
    assert c.post("/logout").status_code == 403
    assert c.get("/api/models").status_code == 200  # leitura não exige


class FakeGoogle:
    def __init__(self, info):
        self.info = info

    def authorize_redirect(self, uri, **kw):
        from flask import redirect
        return redirect("https://accounts.google.com/auth?redirect_uri=" + uri)

    def authorize_access_token(self):
        if self.info is None:
            raise RuntimeError("falha")
        return {"userinfo": self.info}


def test_google_callback(srv):
    c = srv.client(None)
    srv.app.extensions["google"] = FakeGoogle({"email": "Alice@Exemplo.com", "email_verified": True, "name": "Alice"})
    r = c.get("/login/google")
    assert r.status_code == 302 and "accounts.google.com" in r.headers["Location"]
    r = c.get("/auth/callback")
    assert r.status_code == 302 and r.headers["Location"].endswith("/")
    assert c.get("/api/models").status_code == 200
    assert c.post("/logout", headers=H).get_json()["ok"] and c.get("/api/models").status_code == 401


@pytest.mark.parametrize("info,code", [
    ({"email": "intruso@x.com", "email_verified": True}, 403),     # fora da lista
    ({"email": ALICE, "email_verified": False}, 403),              # e-mail não verificado
    (None, 400),                                                   # falha no Google
])
def test_google_callback_rejected(srv, info, code):
    c = srv.client(None)
    srv.app.extensions["google"] = FakeGoogle(info)
    assert c.get("/auth/callback").status_code == code
    assert c.get("/api/models").status_code == 401


# ---------- fluxo de arquivos ----------
def test_full_flow_files_lifecycle(srv, media_dir):
    c = srv.client()
    r = upload(c, media_dir / "a.mp3", name="Minha Aula.mp3")
    assert r.status_code == 202
    jid = r.get_json()["id"]
    s = wait(c, jid)
    assert s["status"] == "done", s
    time.sleep(0.2)
    # upload apagado após o processamento; legenda em download/
    assert srv.upload == [] and srv.download == [f"{jid}.srt"]
    assert s["download_name"] == "Minha Aula.pt-BR.srt" and s["filename"] == "Minha Aula.mp3"
    # a tela pode ser restaurada (reload)
    assert c.get("/api/jobs/current").get_json()["job"]["id"] == jid
    dl = c.get(f"/api/jobs/{jid}/download")
    assert dl.status_code == 200 and b"-->" in dl.data and "Minha Aula.pt-BR.srt" in dl.headers["Content-Disposition"]
    # após o download: legenda apagada, job esquecido, tela zerada
    assert srv.download == [] and srv.upload == []
    assert c.get(f"/api/jobs/{jid}/download").status_code == 404
    assert c.get("/api/jobs/current").get_json()["job"] is None


def test_download_stays_available_until_cleared(srv, media_dir):
    c = srv.client()
    jid = upload(c, media_dir / "a.mp3").get_json()["id"]
    wait(c, jid)
    time.sleep(0.3)
    assert srv.download == [f"{jid}.srt"]                  # continua disponível
    assert c.delete(f"/api/jobs/{jid}", headers=H).status_code == 200   # "outro arquivo escolhido"/Limpar
    assert srv.download == [] and c.get(f"/api/jobs/{jid}").status_code == 404


def test_error_job_removes_all_files_and_can_be_cleared(srv, media_dir):
    c = srv.client()
    jid = upload(c, media_dir / "mudo.mp4", name="mudo.mp4").get_json()["id"]
    s = wait(c, jid)
    assert s["status"] == "error" and "áudio" in s["error"]
    time.sleep(0.2)
    assert srv.upload == [] and srv.download == []
    assert c.delete(f"/api/jobs/{jid}", headers=H).status_code == 200


def test_one_job_per_user_until_cleared(srv, media_dir):
    c = srv.client()
    jid = upload(c, media_dir / "a.mp3").get_json()["id"]
    wait(c, jid)
    time.sleep(0.2)
    r = upload(c, media_dir / "a.mp3")
    assert r.status_code == 409 and srv.upload == []        # nada foi gravado
    c.delete(f"/api/jobs/{jid}", headers=H)
    assert upload(c, media_dir / "a.mp3").status_code == 202


def test_users_are_isolated(srv, media_dir):
    a, b = srv.client(ALICE), srv.client(BOB)
    jid = upload(a, media_dir / "a.mp3").get_json()["id"]
    wait(a, jid)
    assert b.get(f"/api/jobs/{jid}").status_code == 404
    assert b.get(f"/api/jobs/{jid}/download").status_code == 404
    assert b.delete(f"/api/jobs/{jid}", headers=H).status_code == 404
    assert b.post(f"/api/jobs/{jid}/cancel", headers=H).status_code == 404
    assert b.get("/api/jobs/current").get_json()["job"] is None
    assert a.get(f"/api/jobs/{jid}").status_code == 200


def test_single_fifo_queue_and_cancel_queued(tmp_path, media_dir):
    fake = FakeClient()
    gate = threading.Event()
    orig = fake.transcribe

    def slow(model, wav):
        gate.wait(10)
        return orig(model, wav)

    fake.transcribe = slow
    srv = Server(tmp_path, fake)
    a, b = srv.client(ALICE), srv.client(BOB)
    ja = upload(a, media_dir / "a.mp3", name="a.mp3").get_json()["id"]
    time.sleep(1.0)                                         # a começou a rodar
    jb = upload(b, media_dir / "a.mp3", name="b.mp3").get_json()["id"]
    sb = b.get(f"/api/jobs/{jb}").get_json()
    assert a.get(f"/api/jobs/{ja}").get_json()["status"] == "running"
    assert sb["status"] == "queued" and sb["queue_position"] == 1
    assert len(srv.upload) == 2                             # os dois uploads existem
    assert b.post(f"/api/jobs/{jb}/cancel", headers=H).status_code == 200
    assert b.get(f"/api/jobs/{jb}").get_json()["status"] == "cancelled"
    assert len(srv.upload) == 1                             # cancelado na fila: apagado na hora
    gate.set()
    assert wait(a, ja)["status"] == "done"
    time.sleep(0.3)
    assert srv.upload == []


def test_queue_runs_in_arrival_order(tmp_path, media_dir):
    fake = FakeClient()
    gate = threading.Event()
    started = []
    orig = fake.transcribe

    def slow(model, wav):
        gate.wait(10)
        return orig(model, wav)

    fake.transcribe = slow
    srv = Server(tmp_path, fake)
    orig_runner = srv.manager._runner
    srv.manager._runner = lambda job: (started.append(job.owner), orig_runner(job))[1]
    a, b = srv.client(ALICE), srv.client(BOB)
    ja = upload(a, media_dir / "a.mp3").get_json()["id"]
    time.sleep(0.5)
    jb = upload(b, media_dir / "a.mp3").get_json()["id"]
    gate.set()
    wait(a, ja)
    wait(b, jb)
    assert started == [ALICE, BOB]


def test_cannot_clear_while_running(tmp_path, media_dir):
    fake = FakeClient()
    gate = threading.Event()
    orig = fake.transcribe
    fake.transcribe = lambda m, w: (gate.wait(10), orig(m, w))[1]
    srv = Server(tmp_path, fake)
    c = srv.client()
    jid = upload(c, media_dir / "a.mp3").get_json()["id"]
    time.sleep(0.8)
    assert c.delete(f"/api/jobs/{jid}", headers=H).status_code == 409
    c.post(f"/api/jobs/{jid}/cancel", headers=H)
    gate.set()
    assert wait(c, jid)["status"] == "cancelled"
    time.sleep(0.3)
    assert srv.upload == [] and srv.download == []


# ---------- validações e limites ----------
def test_upload_validation(srv, media_dir):
    c = srv.client()
    assert c.post("/api/jobs", headers=H, data={"language": "pt-BR", "model": "fake-audio:1b"}).status_code == 400
    assert upload(c, media_dir / "a.mp3", name="x.exe").status_code == 400
    assert upload(c, media_dir / "a.mp3", language="xx").status_code == 400
    assert upload(c, media_dir / "a.mp3", model="gemma4:e4b").status_code == 400   # catálogo não instalado: recusado
    assert srv.upload == [] and srv.manager.all_jobs() == []


def test_models_only_installed(srv):
    models = srv.client().get("/api/models").get_json()["models"]
    assert models and all(m["installed"] for m in models)


def test_size_limit(tmp_path):
    srv = Server(tmp_path, max_upload_mb=1)
    c = srv.client()
    big = tmp_path / "big.mp3"
    big.write_bytes(b"0" * (2 * 1024 * 1024))
    r = upload(c, big)
    assert r.status_code == 413 and "limite" in r.get_json()["error"]
    assert srv.upload == []


def test_duration_limit(tmp_path, media_dir):
    srv = Server(tmp_path, max_duration_min=0.5)            # 30 s; o áudio de teste tem 50 s
    c = srv.client()
    s = wait(c, upload(c, media_dir / "a.mp3").get_json()["id"])
    assert s["status"] == "error" and "limite" in s["error"]
    time.sleep(0.2)
    assert srv.upload == [] and srv.download == []


def test_ttl_sweeper(srv, media_dir):
    c = srv.client()
    jid = upload(c, media_dir / "a.mp3").get_json()["id"]
    wait(c, jid)
    time.sleep(0.3)
    assert sweep(srv.manager, srv.cfg, now=time.time() + 10) == 0           # dentro do TTL
    assert sweep(srv.manager, srv.cfg, now=time.time() + 61 * 60) == 1      # expirou
    assert srv.download == [] and c.get(f"/api/jobs/{jid}").status_code == 404


def test_startup_purges_leftovers(tmp_path):
    cfg = cfg_for(tmp_path)
    os.makedirs(cfg.upload_dir)
    os.makedirs(cfg.download_dir)
    open(os.path.join(cfg.upload_dir, "velho.mp3"), "w").close()
    open(os.path.join(cfg.download_dir, "velho.srt"), "w").close()
    create_app(Config(), FakeClient(), server=cfg)
    assert os.listdir(cfg.upload_dir) == [] and os.listdir(cfg.download_dir) == []


def test_index_shows_user_and_server_mode(srv):
    html = srv.client().get("/").get_data(as_text=True)
    assert ALICE in html and 'type="file"' in html and "btnBrowse" not in html and "btnSettings" not in html
