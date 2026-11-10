"""Ponto de entrada do executável: sobe o Flask, abre o navegador e encerra o que iniciou."""
import os
import socket
import sys
import threading
import webbrowser

from werkzeug.serving import make_server

from app import create_app


def _bind_port(host: str, preferred: int) -> int:
    for port in (preferred, 0):  # 0 = qualquer porta livre
        with socket.socket() as s:
            try:
                s.bind((host, port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("Nenhuma porta disponível.")


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = _bind_port(host, int(os.environ.get("PORT", 5000)))
    app = create_app(bootstrap=os.environ.get("MINITRANSCRITOR_NO_BOOTSTRAP") != "1")  # NO_BOOTSTRAP: só p/ smoke test
    server = make_server(host, port, app, threaded=True)
    url = f"http://{host}:{port}"
    print(f"Mini Transcritor rodando em {url}")
    print("Feche esta janela (ou pressione Ctrl+C) para encerrar.")
    if os.environ.get("NO_BROWSER") != "1":
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.extensions["setup"].shutdown()  # só para o Ollama se foi este app que o iniciou


if __name__ == "__main__":
    main()
