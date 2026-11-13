"""Teste de compatibilidade Google ADK + Ollama + áudio — RODE SÓ COM A GPU LIVRE, num AMBIENTE VIRTUAL SEPARADO.

    python -m venv .venv-adk && .venv-adk\\Scripts\\activate
    pip install google-adk litellm requests numpy
    python tools/adk_probe.py D:/clips/fragmento.wav --model gemma4:e4b --expect "palavra que aparece no áudio"

Responde, em poucos segundos de GPU, às dúvidas que decidem o desenho do pipeline "Qualidade":
  T1  O agente fala com o Ollama (LiteLlm "ollama_chat/<modelo>") e responde texto?
  T2  Esse modelo CHAMA uma ferramenta simples (tool calling) pelo ADK?
  T3  Ferramenta própria `ouvir_fragmento` (usa o nosso OllamaClient e o WAV): o agente a chama e usa o texto?   <- caminho previsto
  T4  Áudio como parte nativa da mensagem (types.Part inline_data) chega ao modelo? (relatos dizem que é descartado/recusado)
Cada teste imprime OK/FALHOU com o motivo. A API do ADK aqui foi escrita de memória (google.adk.agents.LlmAgent,
google.adk.models.lite_llm.LiteLlm, Runner + InMemorySessionService, google.genai.types): se a versão instalada diferir, ajuste.
"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):  # o console do Windows (cp1252) não imprime acentos/setas
    _s.reconfigure(encoding="utf-8", errors="replace")

APP, USER = "adk_probe", "probe"


def listen_tool_factory(client, model, wav_path):
    def ouvir_fragmento(fragmento: str) -> dict:
        """Ouve o fragmento de áudio indicado e devolve o texto transcrito.

        Args:
            fragmento: identificador do fragmento (qualquer texto; a sonda usa sempre o mesmo arquivo).
        """
        with open(wav_path, "rb") as f:
            return {"texto": client.transcribe(model, f.read())}

    return ouvir_fragmento


def somar(a: int, b: int) -> dict:
    """Soma dois números inteiros.

    Args:
        a: primeiro número.
        b: segundo número.
    """
    return {"resultado": a + b}


async def run_agent(agent, prompt, parts=None):
    """Executa o agente uma vez; devolve (texto final, chamadas de ferramenta)."""
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    sessions = InMemorySessionService()
    await sessions.create_session(app_name=APP, user_id=USER, session_id="s")
    runner = Runner(agent=agent, app_name=APP, session_service=sessions)
    content = types.Content(role="user", parts=[types.Part(text=prompt), *(parts or [])])
    final, calls = "", []
    async for ev in runner.run_async(user_id=USER, session_id="s", new_message=content):
        calls += [fc.name for fc in (ev.get_function_calls() or [])]
        if ev.is_final_response() and ev.content and ev.content.parts:
            final = "".join(p.text or "" for p in ev.content.parts)
    return final, calls


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", help="fragmento curto (5-15 s) em WAV 16 kHz mono")
    ap.add_argument("--model", required=True, help="modelo do Ollama com áudio, ex.: gemma4:e4b")
    ap.add_argument("--expect", default="", help="palavra que deve aparecer na transcrição (confere T3/T4)")
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    args = ap.parse_args()

    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm
    from google.genai import types

    from transcritor.config import normalize_host
    from transcritor.ollama_client import OllamaClient

    host = normalize_host(args.host)
    os.environ["OLLAMA_API_BASE"] = host
    client = OllamaClient(host, 600)
    llm = LiteLlm(model=f"ollama_chat/{args.model}")
    results = []

    def record(name, ok, detail):
        results.append((name, ok))
        print(f"{name}: {'OK' if ok else 'FALHOU'} — {detail}", flush=True)

    def attempt(name, coro_fn, check):
        try:
            final, calls = asyncio.run(coro_fn())
            ok, detail = check(final, calls)
        except Exception as e:  # o objetivo é justamente descobrir o que quebra
            ok, detail = False, f"{e.__class__.__name__}: {str(e)[:300]}"
        record(name, ok, detail)

    attempt("T1 texto simples",
            lambda: run_agent(LlmAgent(name="t1", model=llm, instruction="Responda em uma frase curta."), "Diga olá."),
            lambda f, c: (bool(f.strip()), f"resposta: {f.strip()[:80]!r}"))
    attempt("T2 tool calling",
            lambda: run_agent(LlmAgent(name="t2", model=llm, tools=[somar],
                                       instruction="Use a ferramenta somar para calcular e responda só com o número."), "Quanto é 17 + 25?"),
            lambda f, c: ("somar" in c and "42" in f, f"ferramentas chamadas: {c}; resposta: {f.strip()[:60]!r}"))
    attempt("T3 ferramenta de áudio (nosso OllamaClient)",
            lambda: run_agent(LlmAgent(name="t3", model=llm, tools=[listen_tool_factory(client, args.model, args.wav)],
                                       instruction="Chame ouvir_fragmento e responda com a transcrição devolvida."), "Transcreva o fragmento 1."),
            lambda f, c: ("ouvir_fragmento" in c and (not args.expect or args.expect.lower() in f.lower()),
                          f"ferramentas: {c}; resposta: {f.strip()[:100]!r}"))
    with open(args.wav, "rb") as fh:
        part = types.Part.from_bytes(data=fh.read(), mime_type="audio/wav")
    attempt("T4 áudio nativo na mensagem",
            lambda: run_agent(LlmAgent(name="t4", model=llm, instruction="Transcreva o áudio recebido."), "Transcreva este áudio.", [part]),
            lambda f, c: (bool(f.strip()) and (not args.expect or args.expect.lower() in f.lower()), f"resposta: {f.strip()[:100]!r}"))

    # T4b: mesma mensagem, mas pela rota "openai-compatível" do Ollama (/v1): o LiteLLM aceita `input_audio` nela
    llm_v1 = LiteLlm(model=f"openai/{args.model}", api_base=host + "/v1", api_key="ollama")
    attempt("T4b áudio nativo pela rota /v1 (openai-compatível)",
            lambda: run_agent(LlmAgent(name="t4b", model=llm_v1, instruction="Transcreva o áudio recebido."), "Transcreva este áudio.", [part]),
            lambda f, c: (bool(f.strip()) and (not args.expect or args.expect.lower() in f.lower()), f"resposta: {f.strip()[:100]!r}"))

    print(f"\nResumo: {sum(ok for _, ok in results)}/{len(results)} OK")
    print("Leitura: T3 OK = caminho previsto funciona; T4 OK = o ADK passa áudio direto (melhor ainda); "
          "T2 FALHOU = modelo local sem tool calling confiável (usar só agentes de fluxo com output_key).")


if __name__ == "__main__":
    main()
