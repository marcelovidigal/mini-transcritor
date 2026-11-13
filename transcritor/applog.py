"""Log em arquivo do app (para avaliações futuras e depuração), em vez de despejar ruído técnico na tela.

Fica em `<pasta de dados>/logs/mini-transcritor.log` (Windows: %LOCALAPPDATA%\\MiniTranscritor\\logs\\), rotativo: 5 MB × 5 arquivos.
Cada trabalho registra início, fim (status, tempos por etapa, saída), TODOS os avisos que antes apareciam na tela (revisões que falharam,
repetições removidas, leitura rápida…) e, em erro inesperado, o traceback. Detalhes por fragmento (ex.: motivo de cada revisão que falhou)
vão para o mesmo arquivo em tempo real. O nome do arquivo de mídia é registrado; o conteúdo das falas, nunca.
"""
import logging
import logging.handlers
import os

from .settings import data_dir

LOGGER = "transcritor"
FILE_NAME = "mini-transcritor.log"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def log_dir() -> str:
    return os.path.join(data_dir(), "logs")


def log_path() -> str:
    return os.path.join(log_dir(), FILE_NAME)


def setup(path: str | None = None, level: int = logging.INFO) -> str:
    """Liga o log em arquivo (idempotente). Devolve o caminho do arquivo. Nunca levanta erro: sem permissão de escrita, o app segue sem log."""
    path = path or log_path()
    root = logging.getLogger(LOGGER)
    root.setLevel(level)
    for h in root.handlers:
        if getattr(h, "_mt_file", None) == path:
            return path
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    except OSError:
        return ""
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler._mt_file = path  # marca para não duplicar ao chamar de novo
    root.addHandler(handler)
    return path


def close() -> None:
    """Fecha e remove os handlers de arquivo (usado nos testes para liberar o arquivo no Windows)."""
    root = logging.getLogger(LOGGER)
    for h in list(root.handlers):
        if hasattr(h, "_mt_file"):
            root.removeHandler(h)
            h.close()
