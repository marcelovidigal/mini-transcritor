"""Prova de conceito do pipeline "Qualidade" no Google ADK SEM LiteLLM: o Ollama entra como backend/ferramenta.

    python tools/adk_ollama_agents_probe.py D:/clips/frag/0000.wav --model-a gemma4:e4b --model-b gemma4:e2b

O ADK só orquestra (SequentialAgent + ParallelAgent); os agentes são classes nossas (BaseAgent) que chamam o
OllamaClient do projeto (áudio e texto) e gravam o resultado no estado da sessão (state_delta). Para provar que o
LiteLLM não é necessário, o script o torna não importável (sys.modules["litellm"] = None).
Desenho:  Sequential[ Parallel[ OuvirA, OuvirB ], Juiz ]   ->  estado: hip_a, hip_b, final
RODE SÓ COM A GPU LIVRE, num ambiente com google-adk (ver tools/adk_probe.py). API do ADK escrita de memória: ajuste se preciso.
"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):  # o console do Windows (cp1252) não imprime acentos/setas
    _s.reconfigure(encoding="utf-8", errors="replace")
sys.modules["litellm"] = None  # qualquer "import litellm" passa a falhar: o teste prova que ele não é usado

from collections.abc import AsyncGenerator  # noqa: E402

from google.adk.agents import BaseAgent, ParallelAgent, SequentialAgent  # noqa: E402
from google.adk.agents.invocation_context import InvocationContext  # noqa: E402
from google.adk.events import Event, EventActions  # noqa: E402
from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402

from transcritor.config import normalize_host  # noqa: E402
from transcritor.ollama_client import OllamaClient  # noqa: E402


def _event(author: str, key: str, text: str) -> Event:
    return Event(author=author, content=types.Content(role="model", parts=[types.Part(text=text)]),
                 actions=EventActions(state_delta={key: text}))


class OuvirFragmento(BaseAgent):
    """Gera uma transcrição do fragmento com um modelo de áudio do Ollama e guarda em state[out_key]."""

    model: str
    wav_path: str
    out_key: str
    host: str

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        client = OllamaClient(self.host, 600)
        with open(self.wav_path, "rb") as f:
            wav = f.read()
        text = await asyncio.to_thread(client.transcribe, self.model, wav)  # chamada bloqueante fora do loop
        yield _event(self.name, self.out_key, text)


class Juiz(BaseAgent):
    """Lê as duas hipóteses do estado e pede a um modelo (só texto) a versão final. Aqui, uma tarefa simples de teste."""

    model: str
    host: str

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        a, b = ctx.session.state.get("hip_a", ""), ctx.session.state.get("hip_b", "")
        client = OllamaClient(self.host, 600)
        prompt = ("Duas transcrições do mesmo trecho de áudio. Devolva SOMENTE a melhor versão, sem comentários.\n"
                  f"A: {a}\nB: {b}")
        payload = {"model": self.model, "stream": False, "think": False, "options": {"temperature": 0},
                   "messages": [{"role": "user", "content": prompt}]}
        final = await asyncio.to_thread(client._chat, payload)
        yield _event(self.name, "final", final)


async def main_async(args) -> int:
    host = normalize_host(args.host)
    pipeline = SequentialAgent(name="qualidade", sub_agents=[
        ParallelAgent(name="geracoes", sub_agents=[
            OuvirFragmento(name="ouvir_a", model=args.model_a, wav_path=args.wav, out_key="hip_a", host=host),
            OuvirFragmento(name="ouvir_b", model=args.model_b, wav_path=args.wav, out_key="hip_b", host=host),
        ]),
        Juiz(name="juiz", model=args.model_a, host=host),
    ])
    sessions = InMemorySessionService()
    await sessions.create_session(app_name="qualidade", user_id="u", session_id="s")
    runner = Runner(agent=pipeline, app_name="qualidade", session_service=sessions)
    authors = []
    async for ev in runner.run_async(user_id="u", session_id="s",
                                     new_message=types.Content(role="user", parts=[types.Part(text="processar")])):
        authors.append(ev.author)
    state = (await sessions.get_session(app_name="qualidade", user_id="u", session_id="s")).state
    ok = all(state.get(k) for k in ("hip_a", "hip_b", "final")) and "litellm" not in {m for m in sys.modules if sys.modules[m]}
    print("eventos por agente:", authors)
    for k in ("hip_a", "hip_b", "final"):
        print(f"  {k}: {str(state.get(k))[:140]!r}")
    print("litellm importado:", sys.modules.get("litellm") is not None)
    print("\nRESULTADO:", "OK — ADK orquestrou agentes que usam o Ollama direto, sem LiteLLM" if ok else "FALHOU")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("--model-a", default="gemma4:e4b")
    ap.add_argument("--model-b", default="gemma4:e2b")
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    sys.exit(asyncio.run(main_async(ap.parse_args())))


if __name__ == "__main__":
    main()
