"""Modo servidor (web): login Google, upload → fila única → .srt para download, arquivos temporários.

Fluxo de arquivos:  upload/<id>.<ext>  (apagado ao fim do processamento)
                    download/<id>.srt  (apagado ao baixar, ao limpar a tela ou por TTL)
"""
import io
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field

from flask import Response, jsonify, redirect, render_template_string, request, send_file, session, url_for

from . import engines, media
from .ollama_client import LANGUAGES, OllamaError

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
CSRF_HEADER = ("X-Requested-With", "fetch")


class ConfigError(Exception):
    pass


def _csv(value: str) -> set:
    return {v.strip().lower() for v in (value or "").split(",") if v.strip()}


@dataclass
class ServerConfig:
    secret_key: str
    google_client_id: str
    google_client_secret: str
    allowed_emails: set = field(default_factory=set)
    allowed_domains: set = field(default_factory=set)
    base_url: str = ""  # ex.: https://transcritor.exemplo.com (define o redirect URI do Google)
    data_dir: str = "server_data"
    max_upload_mb: int = 500
    max_duration_min: int = 180
    job_ttl_min: int = 60
    trust_proxy: bool = False

    @property
    def upload_dir(self) -> str:
        return os.path.join(self.data_dir, "upload")

    @property
    def download_dir(self) -> str:
        return os.path.join(self.data_dir, "download")

    @property
    def cookie_secure(self) -> bool:
        return self.base_url.lower().startswith("https://")

    def is_allowed(self, email: str) -> bool:
        email = (email or "").lower()
        return bool(email) and (email in self.allowed_emails or email.rsplit("@", 1)[-1] in self.allowed_domains)

    @classmethod
    def from_env(cls, env=None) -> "ServerConfig":
        env = os.environ if env is None else env
        problems = []
        for var in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET"):
            if not env.get(var):
                problems.append(f"{var} não definido (crie as credenciais OAuth no Google Cloud Console).")
        if len(env.get("SECRET_KEY", "")) < 16:
            problems.append("SECRET_KEY ausente ou curta (mín. 16 caracteres; gere com: python -c \"import secrets;print(secrets.token_hex(32))\").")
        emails, domains = _csv(env.get("ALLOWED_EMAILS")), _csv(env.get("ALLOWED_DOMAINS"))
        if not emails and not domains:
            problems.append("Defina ALLOWED_EMAILS e/ou ALLOWED_DOMAINS (por segurança, sem lista ninguém entra).")

        def num(name, default, minimum=1):
            try:
                v = int(env.get(name, default))
            except ValueError:
                problems.append(f"{name} deve ser um número inteiro.")
                return default
            if v < minimum:
                problems.append(f"{name} deve ser ≥ {minimum}.")
            return v

        cfg = cls(
            secret_key=env.get("SECRET_KEY", ""), google_client_id=env.get("GOOGLE_CLIENT_ID", ""),
            google_client_secret=env.get("GOOGLE_CLIENT_SECRET", ""), allowed_emails=emails, allowed_domains=domains,
            base_url=env.get("BASE_URL", "").rstrip("/"),
            data_dir=env.get("SERVER_DATA_DIR", "server_data"),
            max_upload_mb=num("MAX_UPLOAD_MB", 500), max_duration_min=num("MAX_DURATION_MIN", 180),
            job_ttl_min=num("JOB_TTL_MIN", 60), trust_proxy=env.get("TRUST_PROXY", "0") == "1",
        )
        if problems:
            raise ConfigError("Configuração do modo servidor inválida:\n  - " + "\n  - ".join(problems))
        return cfg


# ---------- arquivos temporários ----------
def _rm(path):
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def purge_dir(path: str) -> None:
    """Recria a pasta vazia (limpeza na inicialização: sobras de execuções anteriores)."""
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def cleanup_after_processing(job) -> None:
    """Ao terminar o processamento: apaga SEMPRE o upload; sem legenda gerada, apaga também a saída."""
    _rm(job.path)
    if job.status != "done":
        _rm(job.output_path)


def remove_job(manager, job) -> None:
    _rm(job.path)
    _rm(job.output_path)
    manager.remove(job.id)


def sweep(manager, cfg: ServerConfig, now: float | None = None) -> int:
    """Remove jobs finalizados há mais que o TTL (e seus arquivos). Devolve quantos foram removidos."""
    now = now or time.time()
    removed = 0
    for job in manager.all_jobs():
        if job.is_finished and job.finished and now - job.finished > cfg.job_ttl_min * 60:
            remove_job(manager, job)
            removed += 1
    return removed


def _start_sweeper(manager, cfg):
    def loop():
        while True:
            time.sleep(60)
            try:
                sweep(manager, cfg)
            except Exception:
                pass
    threading.Thread(target=loop, daemon=True, name="sweeper").start()


_LOGIN_HTML = """<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Entrar — Mini Transcritor</title>
<link rel="stylesheet" href="{{ url_for('static', filename='vendor/bootstrap.min.css') }}"></head>
<body class="bg-light"><main class="container py-5 text-center" style="max-width:420px">
<img src="{{ url_for('static', filename='img/logo.svg') }}" width="72" height="72" alt="Logo">
<h1 class="h4 mt-3">Mini Transcritor</h1>
{% if message %}<div class="alert alert-{{ kind }} mt-3">{{ message }}</div>{% endif %}
<p class="text-body-secondary">Acesso restrito. Entre com sua conta Google autorizada.</p>
<a class="btn btn-primary" href="{{ url_for('login_google') }}">Entrar com Google</a></main></body></html>"""


def register(app, cfg: ServerConfig, manager, client, config, setup, *, version: str) -> None:
    """Registra login, guarda de acesso e as rotas de job do modo servidor."""
    from authlib.integrations.flask_client import OAuth  # só carregado no modo servidor

    os.makedirs(cfg.data_dir, exist_ok=True)
    purge_dir(cfg.upload_dir)
    purge_dir(cfg.download_dir)
    config.max_duration_min = cfg.max_duration_min
    setup.can_auto_install = False
    app.secret_key = cfg.secret_key
    app.config.update(
        MAX_CONTENT_LENGTH=cfg.max_upload_mb * 1024 * 1024 + 4096,
        SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_SECURE=cfg.cookie_secure,
        PERMANENT_SESSION_LIFETIME=12 * 3600,
    )
    oauth = OAuth(app)
    oauth.register(name="google", client_id=cfg.google_client_id, client_secret=cfg.google_client_secret,
                   server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
                   client_kwargs={"scope": "openid email profile"})
    app.extensions["google"] = oauth.google  # os testes substituem por um dublê
    app.extensions["server_config"] = cfg
    _start_sweeper(manager, cfg)
    create_lock = threading.Lock()

    def err(msg, code=400):
        return jsonify({"ok": False, "error": msg}), code

    # ---------- acesso ----------
    public = {"static", "login_page", "login_google", "auth_callback", "healthz"}

    @app.before_request
    def guard():
        ep = request.endpoint
        if ep is None or ep in public:
            return None
        if "user" not in session:
            if request.path.startswith("/api/"):
                return err("Faça login para continuar.", 401)
            return redirect(url_for("login_page"))
        if request.method not in SAFE_METHODS and request.headers.get(CSRF_HEADER[0]) != CSRF_HEADER[1]:
            return err("Requisição recusada (cabeçalho de segurança ausente).", 403)
        return None

    @app.errorhandler(413)
    def too_large(_e):
        return err(f"Arquivo maior que o limite de {cfg.max_upload_mb} MB.", 413)

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True, "version": version})

    @app.get("/login")
    def login_page():
        if "user" in session:
            return redirect("/")
        return render_template_string(_LOGIN_HTML, message=None, kind="info")

    @app.get("/login/google")
    def login_google():
        redirect_uri = (cfg.base_url + "/auth/callback") if cfg.base_url else url_for("auth_callback", _external=True)
        return app.extensions["google"].authorize_redirect(redirect_uri, prompt="select_account")

    @app.get("/auth/callback")
    def auth_callback():
        try:
            token = app.extensions["google"].authorize_access_token()
            info = token.get("userinfo") or {}
        except Exception:
            return Response(render_template_string(_LOGIN_HTML, message="Falha no login com o Google. Tente novamente.",
                                                   kind="danger"), 400)
        email = (info.get("email") or "").lower()
        if not info.get("email_verified") or not cfg.is_allowed(email):
            session.clear()
            return Response(render_template_string(
                _LOGIN_HTML, message=f"A conta {email or 'informada'} não tem acesso a este serviço.", kind="danger"), 403)
        session.clear()
        session.permanent = True
        session["user"] = {"email": email, "name": info.get("name") or email}
        return redirect("/")

    @app.post("/logout")
    def logout():
        session.clear()
        return jsonify({"ok": True})

    # ---------- jobs (sempre restritos ao dono) ----------
    def me() -> str:
        return session["user"]["email"]

    def own(job_id):
        job = manager.get(job_id)
        return job if job and job.owner == me() else None

    @app.post("/api/jobs")
    def create_job():
        language = request.form.get("language", "")
        model = request.form.get("model", "")
        engine_id = request.form.get("engine") or engines.OLLAMA
        diarize = request.form.get("diarize") in ("1", "true", "on")
        context = (request.form.get("context") or "").strip()
        if len(context) > 500:
            return err("O contexto/glossário pode ter no máximo 500 caracteres.")
        f = request.files.get("file")
        if setup.snapshot()["status"] != "ready":
            return err("O servidor ainda está se preparando. Aguarde alguns instantes.", 503)
        if not f or not f.filename:
            return err("Selecione um arquivo de áudio ou vídeo.")
        if not media.is_media_file(f.filename):
            return err("Formato não suportado. Envie um arquivo de áudio ou vídeo (mp4, mkv, mp3, wav, flac…).")
        if language not in LANGUAGES:
            return err("Selecione um idioma de saída válido.")
        if engine_id != engines.OLLAMA:  # motor alternativo: só modelos já baixados no servidor
            msg = engines.check_request(engine_id, model, diarize, config.hf_token, installed_only=True)
            if msg:
                return err(msg)
        else:
            if diarize:
                return err("O Ollama não faz diarização. Use o WhisperX.")
            try:
                installed = {m["name"] for m in client.audio_models()}
            except OllamaError as e:
                return err(str(e), 503)
            if model not in installed:  # no servidor só os modelos já instalados (sem downloads por usuários)
                return err("O modelo selecionado não está disponível neste servidor.")
        with create_lock:  # uma legenda por usuário: processando, na fila ou aguardando download
            if manager.jobs_of(me()):
                return err("Você já tem um arquivo em andamento ou uma legenda pronta. Baixe-a ou clique em Limpar "
                           "antes de enviar outro.", 409)
            job_id = uuid.uuid4().hex[:12]
            os.makedirs(cfg.upload_dir, exist_ok=True)
            path = os.path.join(cfg.upload_dir, job_id + os.path.splitext(f.filename)[1].lower())
            try:
                f.save(path)
            except OSError:
                _rm(path)
                return err("Falha ao gravar o arquivo enviado.", 500)
            job = manager.create(path, language, model, owner=me(), source_name=f.filename,
                                 output_dir=cfg.download_dir, job_id=job_id, engine=engine_id, diarize=diarize, context=context)
        return jsonify({"ok": True, "id": job.id}), 202

    @app.get("/api/jobs/current")
    def current_job():
        jobs = manager.jobs_of(me())
        return jsonify({"ok": True, "job": manager.describe(jobs[0]) if jobs else None})

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id):
        job = own(job_id)
        return jsonify(manager.describe(job)) if job else err("Job não encontrado.", 404)

    @app.post("/api/jobs/<job_id>/cancel")
    def job_cancel(job_id):
        return jsonify({"ok": True}) if own(job_id) and manager.cancel(job_id) else err("Job não encontrado.", 404)

    @app.delete("/api/jobs/<job_id>")
    def job_clear(job_id):
        """Zera a tela: apaga legenda/arquivos do job (só depois de finalizado)."""
        job = own(job_id)
        if not job:
            return err("Job não encontrado.", 404)
        if not job.is_finished:
            return err("Cancele o processamento antes de limpar.", 409)
        remove_job(manager, job)
        return jsonify({"ok": True})

    @app.get("/api/jobs/<job_id>/download")
    def job_download(job_id):
        job = own(job_id)
        if not job or job.status != "done" or not job.output_path or not os.path.isfile(job.output_path):
            return err("Legenda não disponível.", 404)
        with open(job.output_path, "rb") as fh:
            data = fh.read()
        name = job.download_name or "legenda.srt"
        remove_job(manager, job)  # legenda é apagada assim que o download é servido
        return send_file(io.BytesIO(data), as_attachment=True, download_name=name, mimetype="application/x-subrip")
