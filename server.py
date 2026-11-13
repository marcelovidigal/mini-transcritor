"""Entrada do MODO SERVIDOR (web): `python server.py`  (ou via Docker, ver docs/servidor.md).

O modo local/desktop continua sendo `python app.py` / MiniTranscritor.exe. A escolha é só o ponto de entrada.
"""
import os
import sys

from app import create_app
from transcritor import envfile
from transcritor.webserver import ConfigError, ServerConfig


def main() -> int:
    # lê o .env da pasta do projeto (credenciais Google etc.); o que já estiver no ambiente tem prioridade
    envfile.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    try:
        cfg = ServerConfig.from_env()
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 2
    app = create_app(bootstrap=True, server=cfg)
    if cfg.trust_proxy:  # atrás de Caddy/nginx: respeita X-Forwarded-* (https, host, IP)
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
    import waitress

    host, port = os.environ.get("HOST", "0.0.0.0"), int(os.environ.get("PORT", 8000))
    print(f"Mini Transcritor (modo servidor) em http://{host}:{port}  ·  {len(cfg.allowed_emails)} e-mail(s), "
          f"{len(cfg.allowed_domains)} domínio(s) autorizados")
    # Estado (fila/jobs) fica na memória: um único processo, várias threads.
    waitress.serve(app, host=host, port=port, threads=8, max_request_body_size=(cfg.max_upload_mb + 1) * 1024 * 1024)
    return 0


if __name__ == "__main__":
    sys.exit(main())
