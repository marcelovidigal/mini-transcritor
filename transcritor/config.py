import os
from dataclasses import dataclass, field


def normalize_host(host: str) -> str:
    host = (host or "").strip().rstrip("/")
    if not host:
        host = "http://127.0.0.1:11434"
    if not host.startswith(("http://", "https://")):
        host = "http://" + host  # OLLAMA_HOST costuma vir sem esquema (ex.: 0.0.0.0:11434)
    return host.replace("//0.0.0.0", "//127.0.0.1")


@dataclass
class Config:
    """Configuração em runtime (derivada das settings; pode ser alterada pela tela de configurações)."""

    ollama_host: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    chunk_seconds: float = 12.0
    max_chunk_seconds: float = 28.0
    max_duration_min: float = 0  # 0 = sem limite (modo servidor define)
    ollama_timeout: float = field(default_factory=lambda: float(os.environ.get("OLLAMA_TIMEOUT", 600)))
    output_fallback_dir: str = field(
        default_factory=lambda: os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    )

    def __post_init__(self):
        self.ollama_host = normalize_host(self.ollama_host)
