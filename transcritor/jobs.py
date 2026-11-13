"""Estado dos jobs e gerenciador com fila FIFO única (um job processado por vez)."""
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field


class Cancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    path: str
    language: str
    model: str
    engine: str = "ollama"  # ollama | openai-whisper | faster-whisper | whisperx
    diarize: bool = False  # identificar quem fala (só motores com suporte)
    context: str = ""  # tema/nomes/termos que ajudam o modelo (glossário opcional)
    status: str = "queued"  # queued | running | done | error | cancelled
    progress: float = 0.0
    stage: str = "Na fila"
    message: str = ""
    warnings: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)  # segundos por etapa (modelo, extracao, vad, transcricao, traducao, montagem, total)
    audio_seconds: float = 0.0
    work_dir: str | None = None  # pasta dos fragmentos (opção Qualidade); apagada ao concluir ou dar erro
    source_cues: list = field(default_factory=list)  # (início, fim, texto) na língua falada, antes da tradução (avaliação)
    report: dict = field(default_factory=dict)  # relatório do motor Qualidade (fragmentos revisados, origem do texto…)
    output_path: str | None = None
    error: str | None = None
    # modo servidor
    owner: str | None = None  # e-mail da conta Google dona do job
    source_name: str | None = None  # nome original do arquivo enviado
    output_dir: str | None = None  # pasta `download/`
    download_name: str | None = None  # nome sugerido ao baixar o .srt
    created: float = field(default_factory=time.time)
    finished: float | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    winding_down: bool = False  # cancelado pelo usuário, mas o worker ainda está saindo de uma chamada longa

    def update(self, progress: float | None = None, stage: str | None = None, message: str | None = None):
        if progress is not None:
            self.progress = round(max(self.progress, min(100.0, progress)), 1)  # monotônico
        if stage is not None:
            self.stage = stage
        if message is not None:
            self.message = message

    def check_cancel(self):
        if self.cancel_event.is_set():
            raise Cancelled()

    @property
    def is_finished(self) -> bool:
        return self.status in ("done", "error", "cancelled")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "message": self.message,
            "warnings": list(self.warnings),
            "timings": dict(self.timings),
            "audio_seconds": self.audio_seconds,
            "report": dict(self.report),
            "output_path": self.output_path,
            "error": self.error,
            "filename": self.source_name,
            "engine": self.engine,
            "download_name": self.download_name,
        }


class JobManager:
    def __init__(self, runner, on_finish=None, on_cancel=None):
        """`runner(job)` executa o pipeline; `on_finish(job)` roda ao terminar qualquer job (ex.: limpar arquivos);
        `on_cancel(job)` roda assim que o usuário cancela um job EM EXECUÇÃO (ex.: cortar as chamadas em andamento ao Ollama).

        Um único worker consome a fila em ordem de chegada (modelos locais são pesados)."""
        self._runner = runner
        self._on_finish = on_finish
        self._on_cancel = on_cancel
        self._jobs: dict[str, Job] = {}
        self._pending: list[str] = []  # ids na fila, em ordem
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._loop, daemon=True, name="job-worker").start()

    # --- API ------------------------------------------------------------
    def create(self, path: str, language: str, model: str, *, owner=None, source_name=None, output_dir=None,
               job_id: str | None = None, engine: str = "ollama", diarize: bool = False, context: str = "") -> Job:
        job = Job(id=job_id or uuid.uuid4().hex[:12], path=path, language=language, model=model,
                  engine=engine, diarize=diarize, context=context, owner=owner, source_name=source_name, output_dir=output_dir)
        with self._lock:
            self._jobs[job.id] = job
            self._pending.append(job.id)
        self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def jobs_of(self, owner: str) -> list[Job]:
        with self._lock:
            return [j for j in self._jobs.values() if j.owner == owner]

    def all_jobs(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def remove(self, job_id: str) -> Job | None:
        with self._lock:
            if job_id in self._pending:
                self._pending.remove(job_id)
            return self._jobs.pop(job_id, None)

    def describe(self, job: Job) -> dict:
        """to_dict + posição na fila (1 = próximo a ser processado)."""
        d = job.to_dict()
        with self._lock:
            pos = self._pending.index(job.id) + 1 if job.id in self._pending else 0
            waiting_cancel = pos and any(j.winding_down for j in self._jobs.values())
        d["queue_position"] = pos
        if waiting_cancel:
            d["stage"] = "Aguardando o cancelamento anterior terminar"
        return d

    def cancel(self, job_id: str) -> bool:
        """Cancela. Na fila: sai na hora. Em execução: o job passa a "cancelado" IMEDIATAMENTE (a tela já pode se limpar), as chamadas
        em andamento são cortadas (`on_cancel`) e o worker termina de sair em segundo plano; o próximo job espera por ele."""
        job = self.get(job_id)
        if not job:
            return False
        job.cancel_event.set()
        with self._lock:
            was_queued = job.status == "queued" and job_id in self._pending
            was_running = job.status == "running"
            if was_queued:
                self._pending.remove(job_id)
            if was_queued or was_running:
                job.status, job.stage = "cancelled", "Cancelado"
                job.finished = time.time()
                job.winding_down = was_running
        if was_queued:
            self._finish(job)  # não vai passar pelo worker: limpa agora
        elif was_running and self._on_cancel:
            try:
                self._on_cancel(job)
            except Exception:  # cortar chamadas nunca impede o cancelamento
                pass
        return True

    # --- worker ---------------------------------------------------------
    def _loop(self):
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None:
                continue
            with self._lock:
                if job.status != "queued":  # cancelado/removido enquanto esperava
                    continue
                if job_id in self._pending:
                    self._pending.remove(job_id)
                job.status = "running"
            self._run(job)

    def _run(self, job: Job):
        try:
            self._runner(job)
            if not job.cancel_event.is_set():
                job.status, job.stage = "done", "Concluído"
                job.update(progress=100)
        except Cancelled:
            job.status, job.stage = "cancelled", "Cancelado"
        except Exception as e:  # erros de domínio têm mensagem amigável
            if job.cancel_event.is_set():  # o erro é consequência do corte das chamadas: foi cancelamento
                job.status, job.stage = "cancelled", "Cancelado"
            else:
                job.status, job.stage = "error", "Erro"
                job.error = str(e) or e.__class__.__name__
        if job.cancel_event.is_set():
            job.status, job.stage = "cancelled", "Cancelado"
        job.finished = job.finished or time.time()
        job.winding_down = False
        self._finish(job)

    def _finish(self, job: Job):
        if self._on_finish:
            try:
                self._on_finish(job)
            except Exception:  # limpeza nunca derruba o worker
                pass
