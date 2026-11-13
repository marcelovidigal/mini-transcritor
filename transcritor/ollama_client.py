"""Cliente HTTP mínimo do Ollama (modelos locais, capability de áudio e transcrição)."""
import base64
import http.client
import json
import os
import socket
import threading
import urllib.parse

import requests

from . import prompts
from .jobs import Cancelled

LANGUAGES = {
    "pt-BR": "Português (Brasil)",
    "pt-PT": "Português (Portugal)",
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "it": "Italiano",
    "ja": "日本語",
    "zh": "中文",
    "ko": "한국어",
    "ru": "Русский",
}
_LANG_PROMPT_NAMES = {
    "pt-BR": "Brazilian Portuguese", "pt-PT": "European Portuguese", "en": "English", "es": "Spanish",
    "fr": "French", "de": "German", "it": "Italian", "ja": "Japanese", "zh": "Chinese",
    "ko": "Korean", "ru": "Russian",
}


class OllamaError(Exception):
    """Erro de comunicação com o servidor Ollama ou de resposta do modelo."""


def _context_note(context: str, verb: str) -> str:
    context = " ".join((context or "").split())[:500]
    return prompts.render("context_note", verb=verb, context=context) if context else ""


def build_prompt(context: str = "", feedback: str = "") -> str:
    """Prompt de transcrição: idioma original, sem comentários (prompts/transcribe.md)."""
    return prompts.render("transcribe", context=_context_note(context, "use these exact spellings when you hear them"), feedback=feedback)


def build_review_prompt(a: str, b: str, context: str = "", c: str = "", notes: bool = False, c_is_previous: bool = False) -> str:
    """Prompt do revisor (prompts/review.md); `notes` pede crítica e termos (laço de refinamento)."""
    names = "A, B or C" if c else "A or B"
    return prompts.render(
        "review", count="Three" if c else "Two", names=names, labels=names,
        notes=prompts.load("review_notes") if notes else "",
        previous=prompts.load("review_previous") if c and c_is_previous else "",
        context=_context_note(context, "use these exact spellings when you hear them"),
        a=a, b=b, c_line=f"\nC: {c}" if c else "")


def build_review_v2_prompt(a: str, b: str, diffs: list[dict] | None, seconds: float, context: str = "", before: str = "",
                           after: str = "", notes: bool = False, trusted: bool = False) -> str:
    """Prompt do revisor da v2 (prompts/review_v2.md, ou review_v2_silence.md quando o B não ouviu fala): raciocínio em etapas,
    uma escolha por ponto de divergência e correções de palavras que os dois erraram."""
    neighbours = prompts.render("review_neighbours", before=before, after=after) if (before or after) else ""
    # tom do prompt: com o A confiável neste arquivo o revisor é NEUTRO; senão o B é a âncora (no filme, A inventava texto)
    common = dict(seconds=f"{seconds:.0f}", neighbours=neighbours, stance=prompts.load("review_stance_neutral" if trusted else "review_stance_anchor"), context=_context_note(context, "use these exact spellings when you hear them"),
                  notes=prompts.load("review_v2_notes") if notes else "")
    if not b:
        return prompts.render("review_v2_silence", a=a, **common)
    if diffs is None:  # A e B divergem demais para listar: o revisor só confere o B (picks vazio) e pode corrigir palavras
        lines = "(A and B differ too much to list: judge B on its own, return an empty list for picks, and use fixes where needed)"
    else:
        lines = "\n".join(f'{k}. A: "{d["a"] or "(nothing)"}" | B: "{d["b"] or "(nothing)"}"' for k, d in enumerate(diffs, 1)) or "(none)"
    return prompts.render("review_v2", a=a, b=b, differences=lines, **common)


def build_refine_prompt(points: list[dict] | None = None, critique: str = "", terms: list[str] | None = None, context: str = "") -> str:
    """Prompt de transcrição da passada de refino da v2 (prompts/transcribe_refine.md): `points` = pontos em disputa SEM veredito;
    `critique` (opcional) = o que o revisor ouviu de diferente; `terms` = nomes e termos. Raciocínio (analysis) antes da transcrição."""
    pts = "; ".join(f'"{d.get("a") or "(nothing)"}" or "{d.get("b") or "(nothing)"}"' for d in (points or [])[:10])
    return prompts.render(
        "transcribe_refine",
        points=prompts.render("feedback_points", points=pts) if pts else "",
        critique=prompts.render("feedback_critique", critique=critique) if critique else "",
        terms=prompts.render("feedback_terms", terms=", ".join((terms or [])[:12])) if terms else "",
        context=_context_note(context, "use these exact spellings when you hear them"))


def build_self_review_prompt(text: str, seconds: float, context: str = "", before: str = "", after: str = "",
                             doubts: list[dict] | None = None, unstable: bool = False) -> str:
    """Prompt do revisor do autorrefinamento (prompts/review_self.md): UMA transcrição, sem segundo gerador (modo de qualidade 1).
    `doubts` = pontos em que as hipóteses da votação discordaram ([{"current","other"}]); o revisor escolhe um lado em cada."""
    neighbours = prompts.render("review_neighbours", before=before, after=after) if (before or after) else ""
    pts = "\n".join(f'{k}. current: "{d["current"] or "(nothing)"}" | other: "{d["other"] or "(nothing)"}"' for k, d in enumerate(doubts or [], 1))
    note = prompts.render("review_self_doubts", unstable=" (they differed a lot, which often means there is no clear speech)" if unstable else "",
                          points=pts) if doubts else ""
    return prompts.render("review_self", seconds=f"{seconds:.0f}", text=text, neighbours=neighbours, doubts=note,
                          context=_context_note(context, "use these exact spellings when you hear them"))


def build_variant_prompt(kind: str, context: str = "") -> str:
    """Prompt de uma hipótese da votação (prompts/transcribe_literal.md ou transcribe_careful.md)."""
    name = {"literal": "transcribe_literal", "careful": "transcribe_careful"}[kind]
    return prompts.render(name, context=_context_note(context, "use these exact spellings when you hear them"))


def build_self_refine_prompt(previous: str, critique: str, terms: list[str] | None = None, context: str = "") -> str:
    """Prompt do gerador no autorrefinamento (prompts/transcribe_self.md): a tentativa anterior + a crítica do revisor."""
    crit = (critique or "(nothing specific)").strip()
    if crit[-1] not in ".!?":
        crit += "."
    return prompts.render("transcribe_self", previous=previous, critique=crit,
                          terms=prompts.render("feedback_terms", terms=", ".join((terms or [])[:12])) if terms else "",
                          context=_context_note(context, "use these exact spellings when you hear them"))


def build_triage_prompt(items: list[dict], context: str = "", limit: int = 20) -> str:
    """Triagem do documento (só texto), prompts/triage.md: `items` = [{"n": 1, "B": ..., "A": ...}]."""
    return prompts.render("triage", n=len(items), limit=limit, context=_context_note(context, "keep these exact spellings"),
                          items=json.dumps(items, ensure_ascii=False))


def build_feedback_note(previous: str, critique: str = "", terms: list[str] | None = None) -> str:
    """Trecho extra do prompt de transcrição nas passadas de refinamento: a versão final anterior e as críticas do revisor."""
    parts = []
    if previous:
        parts.append(prompts.render("feedback_previous", previous=previous))
    if critique:
        parts.append(prompts.render("feedback_critique", critique=critique))
    if terms:
        parts.append(prompts.render("feedback_terms", terms=", ".join(terms[:12])))
    if not parts:
        return ""
    return "".join(parts) + prompts.load("feedback_closing")


def build_polish_prompt(language: str, texts: list[str], glossary: str = "") -> str:
    """Especialista de pontuação: só pontuação, maiúsculas e acentos; as palavras não mudam (prompts/polish.md)."""
    return prompts.render("polish", n=len(texts), language=language or "same as the text",
                          context=_context_note(glossary, "keep these exact spellings"),
                          fragments=json.dumps(texts, ensure_ascii=False))


def build_translation_review_prompt(language: str, pairs: list[tuple[str, str]], glossary: str = "") -> str:
    """Especialista de tradução (prompts/translation_review.md)."""
    items = [{"source": s, "translation": t} for s, t in pairs]
    return prompts.render("translation_review", n=len(pairs), lang=_LANG_PROMPT_NAMES.get(language, language),
                          context=_context_note(glossary, "keep names and terms as given"),
                          items=json.dumps(items, ensure_ascii=False))


def build_translate_prompt(language: str, text: str, context: str = "") -> str:
    return prompts.render("translate", lang=_LANG_PROMPT_NAMES.get(language, language),
                          context=_context_note(context, "keep names and terms as given"), text=text)


def build_batch_translate_prompt(language: str, texts: list[str], context: str = "") -> str:
    return prompts.render("translate_batch", n=len(texts), lang=_LANG_PROMPT_NAMES.get(language, language),
                          context=_context_note(context, "keep names and terms as given"),
                          segments=json.dumps(texts, ensure_ascii=False))


def parse_batch(raw: str, expected: int) -> list[str]:
    """Extrai a lista JSON de strings da resposta do modelo; OllamaError se não tiver `expected` itens."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s[:4].lower() == "json":
            s = s[4:]
    try:
        data = json.loads(s)
    except ValueError:
        raise OllamaError("Resposta de tradução em lote não é JSON válido.")
    if isinstance(data, dict) and len(data) == 1:
        data = next(iter(data.values()))
    if not isinstance(data, list) or len(data) != expected or not all(isinstance(x, str) for x in data):
        raise OllamaError(f"Resposta de tradução em lote com formato inesperado (esperados {expected} itens).")
    return data


# Limite de tokens das respostas em JSON dos agentes (raciocínio curto + decisão). Sem limite, um modelo que não fecha o texto do campo
# `analysis` gera até estourar o tempo (15 min por chamada travada); com o limite a resposta vem truncada, o JSON é inválido e o
# agente trata como falha daquele fragmento (mantém o texto anterior).
MAX_JSON_TOKENS = 768
# Os esquemas JSON limitam o TAMANHO de cada campo (maxLength/maxItems): o Ollama gera com gramática, então a resposta SEMPRE fecha o JSON
# dentro do limite de tokens, mesmo quando o modelo ignora o "máximo de 80 palavras" do prompt (antes, ~7 % das revisões vinham truncadas).
def _str(n: int) -> dict:
    return {"type": "string", "maxLength": n}


S_ANALYSIS, S_CRITIQUE, S_TEXT = _str(600), _str(300), _str(1500)
S_TERMS = {"type": "array", "maxItems": 10, "items": _str(60)}
S_FIXES = {"type": "array", "maxItems": 8, "items": {"type": "object", "properties": {"from": _str(80), "to": _str(80)}, "required": ["from", "to"]}}
NUM_CTX = 32768  # janela de contexto pedida em TODAS as chamadas (mesmo valor = o Ollama não recarrega o modelo entre uma chamada e outra)
TOKENS_PER_SECOND = 12  # teto de geração por segundo de áudio: fala rápida em português ≈ 7 tokens/s
MAX_TRIAGE_TOKENS = 1536
MAX_VARIANT_TOKENS = 512  # hipóteses da votação: um fragmento de até 28 s cabe folgado


def text_cap(*texts) -> int:
    """Teto de tokens para uma resposta em texto (tradução/pontuação): ~1 token por caractere do que entra + folga, no máximo 4096."""
    return int(min(sum(len(t) for t in texts) + 256, 4096))


def token_cap(wav_bytes: bytes) -> int:
    """Teto de tokens gerados para um trecho de áudio WAV (16 kHz, 16 bits, mono): TOKENS_PER_SECOND por segundo + folga, entre 128 e 1024."""
    seconds = max(len(wav_bytes) - 44, 0) / 32000.0
    return int(min(max(seconds * TOKENS_PER_SECOND + 96, 128), 1024))


class _Call:
    """Uma chamada HTTP em andamento ao Ollama (para poder ser cortada pelo cancelamento)."""

    def __init__(self):
        self.conn = None
        self.wake = threading.Event()  # acorda quem espera: a chamada terminou OU foi cortada
        self.aborted = False
        self.status = 0
        self.raw = b""
        self.error = None


class OllamaClient:
    def __init__(self, host: str, timeout: float = 600, keep_alive: str | None = None):
        self.host = host.rstrip("/")
        self.timeout = timeout
        # mantém o modelo na memória entre trechos/trabalhos (padrão do Ollama: 5 min)
        self.keep_alive = keep_alive or os.environ.get("MT_KEEP_ALIVE", "30m")
        try:
            self.num_ctx = int(os.environ.get("MT_NUM_CTX", NUM_CTX))
        except ValueError:
            self.num_ctx = NUM_CTX
        self._active: set = set()  # conexões em andamento (para o cancelamento imediato)
        self._active_lock = threading.Lock()

    def abort_all(self) -> int:
        """Corta AGORA as chamadas em andamento (cancelamento): quem espera por elas recebe `Cancelled` na hora (não depende do sistema
        operacional acordar um socket bloqueado) e o socket é fechado em seguida, para o Ollama ver o cliente sair e parar de gerar."""
        with self._active_lock:
            calls = list(self._active)
        for c in calls:
            c.aborted = True
            c.wake.set()
            conn = c.conn
            if conn is None:  # a thread ainda não abriu a conexão: quando abrir, o resultado é descartado
                continue
            try:
                if conn.sock is not None:
                    conn.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass
        return len(calls)

    def _send(self, call, method: str, path: str, body, timeout) -> None:
        """Faz a chamada HTTP (roda numa thread à parte; o resultado fica em `call`)."""
        u = urllib.parse.urlsplit(self.host)
        https = u.scheme == "https"
        conn = call.conn = (http.client.HTTPSConnection if https else http.client.HTTPConnection)(u.hostname, u.port or (443 if https else 80), timeout=timeout)
        try:
            try:
                conn.connect()
            except (OSError, http.client.HTTPException):  # recusada, fora do ar ou sem resposta ao conectar
                raise OllamaError(f"Não foi possível conectar ao Ollama em {self.host}. O servidor está em execução?")
            conn.request(method, u.path.rstrip("/") + path, body=body, headers={"Content-Type": "application/json"} if body else {})
            resp = conn.getresponse()
            call.status, call.raw = resp.status, resp.read()
        except OllamaError as e:
            call.error = e
        except (socket.timeout, TimeoutError):
            call.error = OllamaError("Tempo esgotado aguardando o Ollama.")
        except (OSError, http.client.HTTPException) as e:
            call.error = OllamaError(f"Erro de comunicação com o Ollama: {e}")
        finally:
            try:
                conn.close()
            except OSError:
                pass
            call.wake.set()

    def _request(self, method: str, path: str, **kw):
        timeout = kw.pop("timeout", self.timeout)
        body = json.dumps(kw["json"]).encode("utf-8") if "json" in kw else None
        call = _Call()
        with self._active_lock:
            self._active.add(call)
        threading.Thread(target=self._send, args=(call, method, path, body, timeout), daemon=True, name="ollama-call").start()
        try:
            call.wake.wait()
        finally:
            with self._active_lock:
                self._active.discard(call)
        if call.aborted:
            raise Cancelled()
        if call.error:
            raise call.error
        if call.status >= 400:
            text = call.raw.decode("utf-8", "replace")
            try:
                detail = json.loads(text).get("error", text)
            except (ValueError, AttributeError):
                detail = text
            raise OllamaError(f"Ollama respondeu {call.status}: {detail}")
        try:
            return json.loads(call.raw)
        except ValueError:
            raise OllamaError("O Ollama devolveu uma resposta que não é JSON.")

    def audio_models(self) -> list[dict]:
        """Modelos locais cuja capability inclui 'audio'."""
        tags = self._request("GET", "/api/tags", timeout=10).get("models", [])
        out = []
        for m in tags:
            name = m.get("name") or m.get("model")
            try:
                info = self._request("POST", "/api/show", json={"model": name}, timeout=15)
            except OllamaError:
                continue
            if "audio" in (info.get("capabilities") or []):
                out.append({"name": name, "size": m.get("size", 0)})
        return sorted(out, key=lambda x: x["size"])

    def pull(self, model: str, progress=None) -> None:
        """Baixa um modelo (streaming). `progress(completed, total, status)` é chamado a cada atualização."""
        import json

        try:
            with requests.post(self.host + "/api/pull", json={"model": model, "stream": True},
                               stream=True, timeout=(10, 180)) as r:
                if r.status_code >= 400:
                    raise OllamaError(f"Ollama respondeu {r.status_code} ao baixar '{model}': {r.text[:200]}")
                for line in r.iter_lines():
                    if not line:
                        continue
                    msg = json.loads(line)
                    if msg.get("error"):
                        raise OllamaError(f"Falha ao baixar '{model}': {msg['error']}")
                    if progress:
                        progress(msg.get("completed", 0), msg.get("total", 0), msg.get("status", ""))
        except requests.RequestException as e:
            raise OllamaError(f"Falha de rede ao baixar '{model}': {e}")

    def unload_all(self) -> int:
        """Descarrega da memória os modelos carregados (keep_alive 0). Útil antes de usar a GPU com outro motor (Whisper na XPU/CUDA)
        e antes de carregar um modelo grande. Devolve quantos foram descarregados; nunca levanta erro."""
        n = 0
        try:
            for m in self._request("GET", "/api/ps", timeout=10).get("models", []):
                self._request("POST", "/api/generate", json={"model": m.get("name") or m.get("model"), "keep_alive": 0}, timeout=60)
                n += 1
        except OllamaError:
            pass
        return n

    def warm_up(self, model: str) -> None:
        """Carrega o modelo na memória sem gerar texto (pode levar dezenas de segundos na 1ª vez)."""
        self._request("POST", "/api/generate", json={"model": model, "keep_alive": self.keep_alive, "options": {"num_ctx": self.num_ctx}})

    def _chat(self, payload: dict) -> str:
        payload.setdefault("keep_alive", self.keep_alive)
        payload.setdefault("options", {}).setdefault("num_ctx", self.num_ctx)
        data = self._request("POST", "/api/chat", json=payload)
        return (data.get("message", {}).get("content") or "").strip()

    def translate(self, model: str, text: str, language: str, context: str = "") -> str:
        """Traduz (ou mantém) `text` para `language` usando o próprio modelo, só com texto."""
        return self._chat({
            "model": model, "stream": False, "think": False, "options": {"temperature": 0, "num_predict": text_cap(text, text)},
            "messages": [{"role": "user", "content": build_translate_prompt(language, text, context)}],
        })

    def translate_batch(self, model: str, texts: list[str], language: str, context: str = "") -> list[str]:
        """Traduz várias falas consecutivas numa só chamada (uma ajuda a contextualizar a outra).

        Devolve uma lista do mesmo tamanho; levanta OllamaError se a resposta não vier nesse formato.
        """
        data = self._chat({
            "model": model, "stream": False, "think": False, "options": {"temperature": 0, "num_predict": text_cap(*texts, *texts)},
            "format": {"type": "array", "items": {"type": "string"}},
            "messages": [{"role": "user", "content": build_batch_translate_prompt(language, texts, context)}],
        })
        return parse_batch(data, len(texts))

    def review(self, model: str, wav_bytes: bytes, a: str, b: str, context: str = "", c: str = "", *,
               notes: bool = False, c_is_previous: bool = False) -> dict:
        """Revisor da opção Qualidade: o modelo REOUVE o fragmento, compara 2 ou 3 transcrições (A, B e opcionalmente C) e devolve
        {"verdict": "A"|"B"|"C"|"merge", "text": ..., "confidence": 0..1}. Com `notes`, devolve também "critique" (texto) e "terms"
        (lista), que alimentam a passada seguinte do laço de refinamento. Levanta OllamaError se a resposta não for JSON válido."""
        verdicts = ["A", "B", "C", "merge"] if c else ["A", "B", "merge"]
        props = {"verdict": {"type": "string", "enum": verdicts}, "text": S_TEXT, "confidence": {"type": "number"}}
        required = ["verdict", "text", "confidence"]
        if notes:
            props["critique"] = S_CRITIQUE
            props["terms"] = S_TERMS
            required += ["critique", "terms"]
        raw = self._chat({
            "model": model, "stream": False, "think": False, "options": {"temperature": 0, "num_predict": MAX_JSON_TOKENS},
            "format": {"type": "object", "properties": props, "required": required},
            "messages": [{"role": "user", "content": build_review_prompt(a, b, context, c, notes, c_is_previous),
                          "images": [base64.b64encode(wav_bytes).decode("ascii")]}],
        })
        try:
            data = json.loads(raw)
        except ValueError:
            raise OllamaError("A revisão não devolveu JSON válido.")
        if not isinstance(data, dict) or "verdict" not in data:
            raise OllamaError("A revisão veio em formato inesperado.")
        return data

    def review_v2(self, model: str, wav_bytes: bytes, a: str, b: str, diffs: list[dict] | None, seconds: float, context: str = "",
                  before: str = "", after: str = "", notes: bool = False, trusted: bool = False) -> dict:
        """Revisor da v2: devolve {"analysis", "speech", "picks": ["A"|"B"...], "fixes": [{"from","to"}], "confidence"
        (+ "critique", "terms" com `notes`)}. O campo `analysis` vem ANTES das decisões no esquema (raciocínio em etapas)."""
        props = {"analysis": S_ANALYSIS, "speech": {"type": "boolean"},
                 "picks": {"type": "array", "maxItems": 40, "items": {"type": "string", "enum": ["A", "B"]}},
                 "fixes": S_FIXES, "confidence": {"type": "number"}}
        if notes:
            props["critique"] = S_CRITIQUE
            props["terms"] = S_TERMS
        raw = self._chat({
            "model": model, "stream": False, "think": False, "options": {"temperature": 0, "num_predict": MAX_JSON_TOKENS},
            "format": {"type": "object", "properties": props, "required": list(props)},
            "messages": [{"role": "user", "content": build_review_v2_prompt(a, b, diffs, seconds, context, before, after, notes, trusted),
                          "images": [base64.b64encode(wav_bytes).decode("ascii")]}],
        })
        try:
            data = json.loads(raw)
        except ValueError:
            raise OllamaError("A revisão não devolveu JSON válido.")
        if not isinstance(data, dict) or "picks" not in data:
            raise OllamaError("A revisão veio em formato inesperado.")
        return data

    def transcribe_reason(self, model: str, wav_bytes: bytes, context: str = "", points: list[dict] | None = None,
                          critique: str = "", terms: list[str] | None = None) -> str:
        """Transcrição da passada de refino da v2, com raciocínio em etapas: JSON {analysis, transcription}; devolve só a transcrição."""
        raw = self._chat({
            "model": model, "stream": False, "think": False, "options": {"temperature": 0, "num_predict": MAX_JSON_TOKENS},
            "format": {"type": "object", "properties": {"analysis": S_ANALYSIS, "transcription": S_TEXT},
                       "required": ["analysis", "transcription"]},
            "messages": [{"role": "user", "content": build_refine_prompt(points, critique, terms, context),
                          "images": [base64.b64encode(wav_bytes).decode("ascii")]}],
        })
        try:
            data = json.loads(raw)
        except ValueError:
            raise OllamaError("A transcrição com raciocínio não devolveu JSON válido.")
        if not isinstance(data, dict) or not isinstance(data.get("transcription"), str):
            raise OllamaError("A transcrição com raciocínio veio em formato inesperado.")
        return data["transcription"]

    def review_self(self, model: str, wav_bytes: bytes, text: str, seconds: float, context: str = "", before: str = "",
                    after: str = "", doubts: list[dict] | None = None, unstable: bool = False) -> dict:
        """Revisor do autorrefinamento: ouve o fragmento e julga UMA transcrição. Devolve {"analysis", "speech", "verdict": "ok"|"fix",
        "fixes": [{"from","to"}], "critique", "terms"} (+ "picks": "current"|"other" por dúvida da votação); o raciocínio (analysis)
        vem antes das decisões."""
        props = {"analysis": S_ANALYSIS, "speech": {"type": "boolean"}}
        if doubts:
            props["picks"] = {"type": "array", "maxItems": 40, "items": {"type": "string", "enum": ["current", "other"]}}
        props.update({"verdict": {"type": "string", "enum": ["ok", "fix"]}, "fixes": S_FIXES, "critique": S_CRITIQUE, "terms": S_TERMS})
        raw = self._chat({
            "model": model, "stream": False, "think": False, "options": {"temperature": 0, "num_predict": MAX_JSON_TOKENS},
            "format": {"type": "object", "properties": props, "required": list(props)},
            "messages": [{"role": "user", "content": build_self_review_prompt(text, seconds, context, before, after, doubts, unstable),
                          "images": [base64.b64encode(wav_bytes).decode("ascii")]}],
        })
        try:
            data = json.loads(raw)
        except ValueError:
            raise OllamaError("A revisão não devolveu JSON válido.")
        if not isinstance(data, dict) or "verdict" not in data:
            raise OllamaError("A revisão veio em formato inesperado.")
        return data

    def transcribe_self(self, model: str, wav_bytes: bytes, context: str = "", previous: str = "", critique: str = "",
                        terms: list[str] | None = None) -> str:
        """Gerador do autorrefinamento: reescuta o fragmento com a tentativa anterior e a crítica do revisor; devolve só a transcrição."""
        raw = self._chat({
            "model": model, "stream": False, "think": False, "options": {"temperature": 0, "num_predict": MAX_JSON_TOKENS},
            "format": {"type": "object", "properties": {"analysis": S_ANALYSIS, "transcription": S_TEXT},
                       "required": ["analysis", "transcription"]},
            "messages": [{"role": "user", "content": build_self_refine_prompt(previous, critique, terms, context),
                          "images": [base64.b64encode(wav_bytes).decode("ascii")]}],
        })
        try:
            data = json.loads(raw)
        except ValueError:
            raise OllamaError("A transcrição com raciocínio não devolveu JSON válido.")
        if not isinstance(data, dict) or not isinstance(data.get("transcription"), str):
            raise OllamaError("A transcrição com raciocínio veio em formato inesperado.")
        return data["transcription"]

    def transcribe_variant(self, model: str, wav_bytes: bytes, context: str = "", kind: str = "literal") -> str:
        """Uma hipótese da votação do modo 1: o mesmo modelo, com outra instrução (`literal` | `careful`); devolve só o texto."""
        return self._chat({  # num_predict: sem fala (música) o modelo pequeno repete texto até estourar o tempo (visto no filme)
            "model": model, "stream": False, "think": False, "options": {"temperature": 0, "num_predict": MAX_VARIANT_TOKENS},
            "messages": [{"role": "user", "content": build_variant_prompt(kind, context),
                          "images": [base64.b64encode(wav_bytes).decode("ascii")]}],
        })

    def triage(self, model: str, items: list[dict], context: str = "", limit: int = 20) -> dict:
        """Triagem do documento (só texto): devolve {"analysis", "glossary": [...], "suspects": [números dos fragmentos]}."""
        raw = self._chat({
            "model": model, "stream": False, "think": False, "options": {"temperature": 0, "num_predict": MAX_TRIAGE_TOKENS},
            "format": {"type": "object", "properties": {"analysis": {"type": "string"},
                                                         "glossary": {"type": "array", "items": {"type": "string"}},
                                                         "suspects": {"type": "array", "items": {"type": "integer"}}},
                       "required": ["analysis", "glossary", "suspects"]},
            "messages": [{"role": "user", "content": build_triage_prompt(items, context, limit)}],
        })
        try:
            data = json.loads(raw)
        except ValueError:
            raise OllamaError("A triagem não devolveu JSON válido.")
        if not isinstance(data, dict) or "suspects" not in data:
            raise OllamaError("A triagem veio em formato inesperado.")
        return data

    def _text_batch(self, model: str, prompt: str, expected: int, *texts: str) -> list[str]:
        raw = self._chat({
            "model": model, "stream": False, "think": False, "options": {"temperature": 0, "num_predict": text_cap(*texts, *texts)},
            "format": {"type": "array", "items": {"type": "string"}},
            "messages": [{"role": "user", "content": prompt}],
        })
        return parse_batch(raw, expected)

    def polish(self, model: str, texts: list[str], language: str = "", glossary: str = "") -> list[str]:
        """Especialista de pontuação (só texto): devolve os mesmos fragmentos com pontuação/maiúsculas ajustadas."""
        return self._text_batch(model, build_polish_prompt(language, texts, glossary), len(texts), *texts)

    def review_translation(self, model: str, pairs: list[tuple[str, str]], language: str, glossary: str = "") -> list[str]:
        """Especialista de tradução (só texto): confere cada (original, tradução) e devolve as traduções finais."""
        return self._text_batch(model, build_translation_review_prompt(language, pairs, glossary), len(pairs), *[x for p in pairs for x in p])

    def transcribe(self, model: str, wav_bytes: bytes, context: str = "", feedback: str = "") -> str:
        """`feedback` (build_feedback_note) entra nas passadas de refinamento: versão anterior + críticas do revisor.
        `num_predict` proporcional à duração do trecho: sem ele, música/ruído fazem o modelo repetir texto por dezenas de milhares de
        tokens (um desses textos, entregue ao revisor, estourou o contexto de 32 mil tokens do Ollama)."""
        return self._chat({
            "model": model,
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_predict": token_cap(wav_bytes)},
            "messages": [
                {
                    "role": "user",
                    "content": build_prompt(context, feedback),
                    "images": [base64.b64encode(wav_bytes).decode("ascii")],
                }
            ],
        })
