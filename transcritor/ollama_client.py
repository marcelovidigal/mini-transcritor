"""Cliente HTTP mínimo do Ollama (modelos locais, capability de áudio e transcrição)."""
import base64

import requests

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


def build_prompt() -> str:
    """Prompt de transcrição: idioma original, sem comentários."""
    return (
        "Transcribe the speech in this audio clip in its original language. "
        "Output ONLY the spoken text as plain text, with normal punctuation. "
        "Do not add comments, labels, timestamps or quotes. "
        "If there is no intelligible speech, output nothing."
    )


def build_translate_prompt(language: str, text: str) -> str:
    lang = _LANG_PROMPT_NAMES.get(language, language)
    return (
        f"Translate the following subtitle text to {lang}. "
        f"If it is already in {lang}, return it unchanged. "
        "Output ONLY the resulting text, with no comments, notes or quotes.\n\n"
        f"Text:\n{text}"
    )


class OllamaClient:
    def __init__(self, host: str, timeout: float = 600):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kw):
        try:
            r = requests.request(method, self.host + path, timeout=kw.pop("timeout", self.timeout), **kw)
        except requests.ConnectionError:
            raise OllamaError(f"Não foi possível conectar ao Ollama em {self.host}. O servidor está em execução?")
        except requests.Timeout:
            raise OllamaError("Tempo esgotado aguardando o Ollama.")
        except requests.RequestException as e:
            raise OllamaError(f"Erro de comunicação com o Ollama: {e}")
        if r.status_code >= 400:
            try:
                detail = r.json().get("error", r.text)
            except ValueError:
                detail = r.text
            raise OllamaError(f"Ollama respondeu {r.status_code}: {detail}")
        return r.json()

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

    def _chat(self, payload: dict) -> str:
        data = self._request("POST", "/api/chat", json=payload)
        return (data.get("message", {}).get("content") or "").strip()

    def translate(self, model: str, text: str, language: str) -> str:
        """Traduz (ou mantém) `text` para `language` usando o próprio modelo, só com texto."""
        return self._chat({
            "model": model, "stream": False, "think": False, "options": {"temperature": 0},
            "messages": [{"role": "user", "content": build_translate_prompt(language, text)}],
        })

    def transcribe(self, model: str, wav_bytes: bytes) -> str:
        return self._chat({
            "model": model,
            "stream": False,
            "think": False,
            "options": {"temperature": 0},
            "messages": [
                {
                    "role": "user",
                    "content": build_prompt(),
                    "images": [base64.b64encode(wav_bytes).decode("ascii")],
                }
            ],
        })
