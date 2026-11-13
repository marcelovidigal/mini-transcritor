"""Mini Transcritor — aplicação Flask.

Dois modos, mesmo código:
  * local (padrão): `python app.py` / executável — seleção de arquivo por caminho, configurações, sem login.
  * servidor: `python server.py` (SERVER_MODE) — upload, login Google, fila única, arquivos temporários.
"""
import os
import threading
import sys

from flask import Flask, jsonify, render_template, request, send_file, session

from transcritor import applog, browse, catalog, engines, media, pipeline, quality_engine
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
    applog.setup()  # log em arquivo (<pasta de dados>/logs/): avisos e detalhes técnicos saem da tela e ficam aqui
    config = config or Config(ollama_host=settings["ollama_host"], chunk_seconds=settings["chunk_seconds"],
                              parallel=settings["parallel"], translate_batch=settings["translate_batch"],
                              hf_token=settings["hf_token"] or os.environ.get("HF_TOKEN", ""),
                              translate_model=settings["base_model"], device=settings["device"])
    client = client or OllamaClient(config.ollama_host, config.ollama_timeout)
    app = Flask(__name__, template_folder=os.path.join(resource_dir(), "templates"),
                static_folder=os.path.join(resource_dir(), "static"))

    on_finish = None
    if server:
        from transcritor import webserver
        on_finish = webserver.cleanup_after_processing  # apaga o upload (e a saída, se falhou) ao terminar cada job
    def on_cancel(job):  # cancelar = cortar AGORA as chamadas em andamento ao Ollama e liberar a GPU (o descarregamento roda à parte)
        if hasattr(client, "abort_all"):
            client.abort_all()
        if hasattr(client, "unload_all"):
            threading.Thread(target=client.unload_all, daemon=True, name="unload-after-cancel").start()

    manager = JobManager(lambda job: pipeline.run_job(job, client, config), on_finish=on_finish, on_cancel=on_cancel)
    setup = SetupManager(settings, client)
    app.extensions.update(job_manager=manager, setup=setup, settings=settings)
    engines.ENGINES["whisper-cpp"].configured_path = settings["whispercpp_path"]
    engines.ENGINES["whisper-cpp"].configured_cuda_path = settings["whispercpp_cuda_path"]
    engines.ENGINES["nemo-speech-cpp"].configured_path = settings["nemocpp_path"]
    engines.ENGINES["nemotron-vllm"].configured_url = settings["nemotron_vllm_url"]
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

    @app.get("/api/engines")
    def engines_list():
        return jsonify({"ok": True, "engines": engines.all_status(), "diarization_ready": bool(config.hf_token)})

    @app.get("/api/devices")
    def devices():
        """Dispositivos para os motores Whisper (CPU, CUDA, Intel XPU, Metal). Só é chamado ao escolher um desses motores."""
        eid = request.args.get("engine", "")
        return jsonify({"ok": True, **engines.devices_status(), "selected": config.device,
                        "plan": engines.engine_plan(eid, config.device) if eid else []})

    @app.get("/api/models")
    def models():
        engine_id = request.args.get("engine", engines.OLLAMA)
        if quality_engine.is_quality(engine_id):
            engine_id = engines.OLLAMA  # a opção Qualidade usa os modelos de áudio do Ollama para a 1ª geração e a revisão
        if engine_id != engines.OLLAMA:  # motores alternativos: catálogo estático; "baixado" só para o que está no disco
            eng = engines.get(engine_id)
            if not eng:
                return jsonify({"ok": False, "models": [], "error": "Motor de transcrição inválido."})
            if not eng.available():
                return jsonify({"ok": False, "models": [], "error": eng.not_installed_message()})
            found = eng.models()
            if server:
                found = [m for m in found if m["installed"]]
            return jsonify({"ok": True, "engine": engine_id, "models": found})
        try:
            merged = catalog.merge(catalog.load(), client.audio_models(), catalog.load_excluded())
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
        data = settings.as_dict()
        data["hf_token_set"] = bool(data.pop("hf_token", ""))  # o token nunca volta para o navegador
        return jsonify({"ok": True, **data, "ffmpeg_active": fm.active_description(),
                        "ffmpeg_local_dir": fm.local_dir()})

    @app.post("/api/settings")
    def save_settings():
        try:
            data = settings.update(request.get_json(silent=True) or {})
        except SettingsError as e:
            return err(str(e))
        config.ollama_host = data["ollama_host"]
        config.chunk_seconds = data["chunk_seconds"]
        config.parallel = data["parallel"]
        config.translate_batch = data["translate_batch"]
        config.hf_token = data["hf_token"] or os.environ.get("HF_TOKEN", "")
        config.translate_model = data["base_model"]
        config.device = data["device"]
        engines.ENGINES["whisper-cpp"].configured_path = data["whispercpp_path"]
        engines.ENGINES["whisper-cpp"].configured_cuda_path = data["whispercpp_cuda_path"]
        engines.ENGINES["nemo-speech-cpp"].configured_path = data["nemocpp_path"]
        engines.ENGINES["nemotron-vllm"].configured_url = data["nemotron_vllm_url"]
        data ={k: v for k, v in data.items() if k != "hf_token"}
        client.host = data["ollama_host"].rstrip("/")
        setup.run()  # reaplica a verificação com as novas configurações
        return jsonify({"ok": True, **data})

    def remember_dir(folder: str) -> None:
        """Guarda a última pasta de trabalho (ponto de partida do navegador de arquivos); nunca atrapalha o fluxo."""
        try:
            if settings["last_dir"] != folder:
                settings.update({"last_dir": folder})
        except Exception:
            pass

    @app.get("/api/browse")
    def browse_dir():
        path = request.args.get("path")
        if request.args.get("start"):  # abertura do diálogo: a pasta pedida, senão a última usada, senão o padrão (unidades/home)
            for cand in (path, settings["last_dir"]):
                if cand and os.path.isdir(cand):
                    path = cand
                    break
            else:
                path = None
        try:
            listing = browse.list_dir(path)
        except (OSError, ValueError) as e:
            return err(f"Não foi possível abrir a pasta: {e.__class__.__name__}")
        if listing["path"]:
            remember_dir(listing["path"])
        return jsonify({"ok": True, **listing})

    @app.post("/api/jobs")
    def create_job():
        data = request.get_json(silent=True) or {}
        path = (data.get("path") or "").strip().strip('"')
        language = data.get("language") or ""
        model = data.get("model") or ""
        engine_id = data.get("engine") or engines.OLLAMA
        diarize = bool(data.get("diarize"))
        context = str(data.get("context") or "").strip()
        if len(context) > 500:
            return err("O contexto/glossário pode ter no máximo 500 caracteres.")
        if not path:
            return err("Informe o caminho do arquivo.")
        if not os.path.isfile(path):
            return err("Arquivo não encontrado. Verifique o caminho informado.")
        remember_dir(os.path.dirname(os.path.abspath(path)))
        if language not in LANGUAGES:
            return err("Selecione um idioma de saída válido.")
        if not model:
            return err("Selecione um modelo.")
        try:
            media.require_ffmpeg()
            if engine_id != engines.OLLAMA and not quality_engine.is_quality(engine_id):
                msg = engines.check_request(engine_id, model, diarize, config.hf_token)
                if msg:
                    return err(msg)
            else:
                if diarize:
                    return err("O Ollama não faz diarização. Use o WhisperX.")
                allowed = {catalog.norm(m["name"]) for m in client.audio_models()}
                allowed |= {catalog.norm(m["name"]) for m in catalog.load()}  # catálogo: baixado sob demanda
                if catalog.norm(model) not in allowed:
                    return err("O modelo selecionado não está disponível ou não suporta áudio.")
        except (media.MediaError, OllamaError) as e:
            return err(str(e))
        job = manager.create(path, language, model, engine=engine_id, diarize=diarize, context=context)
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
