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
    status: str = "queued"  # queued | running | done | error | cancelled
    progress: float = 0.0
    stage: str = "Na fila"
    message: str = ""
    warnings: list = field(default_factory=list)
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
            "output_path": self.output_path,
            "error": self.error,
            "filename": self.source_name,
            "download_name": self.download_name,
        }


class JobManager:
    def __init__(self, runner, on_finish=None):
        """`runner(job)` executa o pipeline; `on_finish(job)` roda ao terminar qualquer job (ex.: limpar arquivos).

        Um único worker consome a fila em ordem de chegada (modelos locais são pesados)."""
        self._runner = runner
        self._on_finish = on_finish
        self._jobs: dict[str, Job] = {}
        self._pending: list[str] = []  # ids na fila, em ordem
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._loop, daemon=True, name="job-worker").start()

    # --- API ------------------------------------------------------------
    def create(self, path: str, language: str, model: str, *, owner=None, source_name=None, output_dir=None,
               job_id: str | None = None) -> Job:
        job = Job(id=job_id or uuid.uuid4().hex[:12], path=path, language=language, model=model,
                  owner=owner, source_name=source_name, output_dir=output_dir)
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
        d["queue_position"] = pos
        return d

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if not job:
            return False
        job.cancel_event.set()
        with self._lock:
            was_queued = job.status == "queued" and job_id in self._pending
            if was_queued:
                self._pending.remove(job_id)
                job.status, job.stage = "cancelled", "Cancelado"
                job.finished = time.time()
        if was_queued:
            self._finish(job)  # não vai passar pelo worker: limpa agora
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
            job.status, job.stage = "done", "Concluído"
            job.update(progress=100)
        except Cancelled:
            job.status, job.stage = "cancelled", "Cancelado"
        except Exception as e:  # erros de domínio têm mensagem amigável
            job.status, job.stage = "error", "Erro"
            job.error = str(e) or e.__class__.__name__
        job.finished = time.time()
        self._finish(job)

    def _finish(self, job: Job):
        if self._on_finish:
            try:
                self._on_finish(job)
            except Exception:  # limpeza nunca derruba o worker
                pass
