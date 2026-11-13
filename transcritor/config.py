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
    parallel: int = 2  # chamadas simultâneas ao Ollama (1–4); ajuste conforme a VRAM / OLLAMA_NUM_PARALLEL
    translate_batch: int = 8  # falas por chamada de tradução (1 = fala a fala)
    device: str = "auto"  # auto | cpu | cuda | xpu | mps (motores Whisper); auto = melhor disponível ou $MT_DEVICE
    hf_token: str = field(default_factory=lambda: os.environ.get("HF_TOKEN", ""))  # diarização (WhisperX)
    translate_model: str = ""  # modelo Ollama que traduz quando o motor não é o Ollama (vazio = automático)
    # motor "Qualidade": 2ª geração, revisor e limiares (ver quality_engine.py)
    quality_b_engine: str = "openai-whisper"
    quality_b_model: str = "large-v3-turbo"
    quality_c_engine: str = ""  # 3ª geração opcional (ex.: openai-whisper); vazio = só A e B
    quality_c_model: str = ""
    quality_reviewer: str = "gemma4:12b"  # revisor (modelo de áudio do Ollama): o maior foi o melhor nos testes; vazio = o mesmo do A
    quality_threshold: float = 0.05
    quality_min_conf: float = 0.5
    quality_prefer: str = "B"  # geração prioritária sem dúvida/revisão: B (Whisper) foi mais confiável que A; "A" = Ollama
    quality_loops: int = 0  # passadas de refinamento depois da 1ª (os modos quality-e4b/-12b usam 2 = 3 passadas no total)
    quality_self: bool = False  # modo de qualidade 1: autorrefinamento gerador <-> revisor, sem WhisperX/gerador B
    quality_gate: bool = False  # modo 1: porteiro de fala (Silero) descarta fragmentos sem fala ANTES de gerar (transcritor/speechgate.py)
    quality_vote: bool = False  # modo 1: 3 hipóteses do mesmo modelo, consenso + dúvidas para o revisor (transcritor/vote.py)
    quality_judge: str = "v1"  # "v2" = B é a âncora, silêncio é informação, decisão por ponto (docs/plano-pipeline-v2.md)
    quality_neighbours: int = 2  # v2: fragmentos vizinhos (antes/depois) mostrados ao revisor como contexto
    quality_feedback: str = "terms"  # v2, passada de refino: "terms" | "points" (+ pontos em disputa) | "critique" (+ crítica do revisor)
    quality_triage: bool = False  # v2: triagem do documento inteiro (só texto) antes da revisão com áudio
    quality_triage_window: int = 60  # v2: fragmentos por chamada da triagem
    quality_specialists: bool = False  # termos + pontuação (antes da legenda) e revisão da tradução (depois dela)
    work_dir: str = ""  # vazio = <projeto>/work
    vad: bool = True  # detecção de fala: pula silêncio e corta nas pausas (False = cortes por tempo)
    max_duration_min: float = 0  # 0 = sem limite (modo servidor define)
    ollama_timeout: float = field(default_factory=lambda: float(os.environ.get("OLLAMA_TIMEOUT", 600)))
    output_fallback_dir: str = field(
        default_factory=lambda: os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    )

    def __post_init__(self):
        self.ollama_host = normalize_host(self.ollama_host)
