"""Mini Transcritor — aplicação Flask.

Dois modos, mesmo código:
  * local (padrão): `python app.py` / executável — seleção de arquivo por caminho, configurações, sem login.
  * servidor: `python server.py` (SERVER_MODE) — upload, login Google, fila única, arquivos temporários.
"""
import os
import sys

from flask import Flask, jsonify, render_template, request, send_file, session

from transcritor import browse, catalog, media, pipeline
from transcritor import ffmpeg_manager as fm
from transcritor.config import Config
from transcritor.jobs import JobManager
from transcritor.ollama_client import LANGUAGES, OllamaClient, OllamaError
from transcritor.settings import Settings, SettingsError
from transcritor.setup import SetupManager
from transcritor.version import __version__


def resource_dir() -> str:
    """Raiz dos templates/estáticos: pasta do PyInstaller quando empacotado, senão a do projeto."""
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))


def create_app(config: Config | None = None, client: OllamaClient | None = None,
               settings: Settings | None = None, bootstrap: bool = False, server=None) -> Flask:
    """`bootstrap=True` executa o preparo (FFmpeg/Ollama/modelo base); sem ele o app já nasce "pronto".

    `server` (ServerConfig) ativa o modo servidor."""
    settings = settings or Settings()
    config = config or Config(ollama_host=settings["ollama_host"], chunk_seconds=settings["chunk_seconds"])
    client = client or OllamaClient(config.ollama_host, config.ollama_timeout)
    app = Flask(__name__, template_folder=os.path.join(resource_dir(), "templates"),
                static_folder=os.path.join(resource_dir(), "static"))

    on_finish = None
    if server:
        from transcritor import webserver
        on_finish = webserver.cleanup_after_processing  # apaga o upload (e a saída, se falhou) ao terminar cada job
    manager = JobManager(lambda job: pipeline.run_job(job, client, config), on_finish=on_finish)
    setup = SetupManager(settings, client)
    app.extensions.update(job_manager=manager, setup=setup, settings=settings)
    if server:
        webserver.register(app, server, manager, client, config, setup, version=__version__)
    if bootstrap:
        setup.run()
    else:
        setup.mark_ready()

    def err(msg: str, code: int = 400):
        return jsonify({"ok": False, "error": msg}), code

    # ---------- rotas comuns aos dois modos ----------
    @app.get("/")
    def index():
        return render_template("index.html", version=__version__, server_mode=bool(server),
                               user=session.get("user") if server else None,
                               max_upload_mb=server.max_upload_mb if server else 0)

    @app.get("/api/setup")
    def setup_status():
        return jsonify(setup.snapshot())

    @app.post("/api/setup/run")
    def setup_run():
        setup.run()
        return jsonify(setup.snapshot())

    @app.get("/api/languages")
    def languages():
        return jsonify([{"code": k, "name": v} for k, v in LANGUAGES.items()])

    @app.get("/api/models")
    def models():
        try:
            merged = catalog.merge(catalog.load(), client.audio_models())
        except OllamaError as e:
            return jsonify({"ok": False, "models": [], "error": str(e)})
        if server:  # no servidor, só modelos já instalados (usuários não disparam downloads)
            merged = [m for m in merged if m["installed"]]
        return jsonify({"ok": True, "models": merged})

    if server:
        return app

    # ---------- rotas só do modo local ----------
    @app.post("/api/setup/install-ollama")
    def setup_install():
        if not setup.install_ollama():
            return err("Instalação automática indisponível neste sistema ou preparo em andamento.")
        return jsonify(setup.snapshot())

    @app.get("/api/settings")
    def get_settings():
        return jsonify({"ok": True, **settings.as_dict(), "ffmpeg_active": fm.active_description(),
                        "ffmpeg_local_dir": fm.local_dir()})

    @app.post("/api/settings")
    def save_settings():
        try:
            data = settings.update(request.get_json(silent=True) or {})
        except SettingsError as e:
            return err(str(e))
        config.ollama_host = data["ollama_host"]
        config.chunk_seconds = data["chunk_seconds"]
        client.host = data["ollama_host"].rstrip("/")
        setup.run()  # reaplica a verificação com as novas configurações
        return jsonify({"ok": True, **data})

    @app.get("/api/browse")
    def browse_dir():
        try:
            return jsonify({"ok": True, **browse.list_dir(request.args.get("path"))})
        except (OSError, ValueError) as e:
            return err(f"Não foi possível abrir a pasta: {e.__class__.__name__}")

    @app.post("/api/jobs")
    def create_job():
        data = request.get_json(silent=True) or {}
        path = (data.get("path") or "").strip().strip('"')
        language = data.get("language") or ""
        model = data.get("model") or ""
        if not path:
            return err("Informe o caminho do arquivo.")
        if not os.path.isfile(path):
            return err("Arquivo não encontrado. Verifique o caminho informado.")
        if language not in LANGUAGES:
            return err("Selecione um idioma de saída válido.")
        if not model:
            return err("Selecione um modelo.")
        try:
            media.require_ffmpeg()
            allowed = {catalog.norm(m["name"]) for m in client.audio_models()}
            allowed |= {catalog.norm(m["name"]) for m in catalog.load()}  # catálogo: baixado sob demanda
            if catalog.norm(model) not in allowed:
                return err("O modelo selecionado não está disponível ou não suporta áudio.")
        except (media.MediaError, OllamaError) as e:
            return err(str(e))
        job = manager.create(path, language, model)
        return jsonify({"ok": True, "id": job.id}), 202

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id):
        job = manager.get(job_id)
        return jsonify(manager.describe(job)) if job else err("Job não encontrado.", 404)

    @app.post("/api/jobs/<job_id>/cancel")
    def job_cancel(job_id):
        return jsonify({"ok": True}) if manager.cancel(job_id) else err("Job não encontrado.", 404)

    @app.get("/api/jobs/<job_id>/download")
    def job_download(job_id):
        job = manager.get(job_id)
        if not job or job.status != "done" or not job.output_path or not os.path.isfile(job.output_path):
            return err("Legenda não disponível.", 404)
        return send_file(job.output_path, as_attachment=True, download_name=os.path.basename(job.output_path),
                         mimetype="application/x-subrip")

    return app


if __name__ == "__main__":
    create_app(bootstrap=True).run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), threaded=True)
