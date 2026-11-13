# Validação e medição (roteiro para quando a GPU estiver livre)

Tudo o que foi implementado nas últimas etapas foi testado com **dublês** (testes automáticos), sem rodar modelos de verdade. Este roteiro valida e mede com os modelos reais. **Rode só com a GPU livre**: usa o Ollama e os motores de verdade.

## Bateria completa (fazer quando a transcrição em andamento terminar)

Ordem sugerida; cada item aponta para a seção que o detalha. Marque ao concluir e anote os números em `docs/modelos.md`.

- [x] **1.** Validar os motores Whisper com `tiny` em cada dispositivo (CPU, XPU) — seção 1 *(OpenAI Whisper CPU e XPU, faster-whisper, whisper.cpp e WhisperX OK; diarização pendente: exige token do Hugging Face)*
- [x] **2.** Conferir o `whisper-cli` (parâmetros reais) e comparar os 4 motores — seção 2 (comparação entre motores) *(ver modelos.md seção 6.2)*
- [x] **3.** Tradução em lote com um modelo real do Ollama (formato JSON, fallback fala a fala) — seção 2 *(OK com e4b, 12b, e2b)*
- [x] **4.** Medir `parallel`, `translate_batch`, VAD, tamanho do trecho e contexto — seção 2 *(modelos.md 6.1; `parallel` sem ganho na Intel Arc)*
- [ ] **5.** Limiar do VAD com fala real e ruído de fundo — seção 2 *(feito só com voz sintética + ruído rosa; folga ajustada para 0,3 s; falta fala real)*
- [x] **6.** **Sonda da opção Qualidade**: divergência entre dois modelos do Ollama — seção 4 *(desfavorável no material sintético; modelos.md 6.3)*
- [x] **7.** **Compatibilidade do Google ADK com Ollama e áudio** — seção 4 *(T1–T3 OK, T4 falha; modelos.md 6.4)*
- [x] **10.** Pipeline completo em áudio de filme (30 min do centro, pt-BR) contra a legenda original — [experimento-longa-metragem.md](experimento-longa-metragem.md); scripts em `tools/experimento_filme/`
- [x] **9.** Matriz da opção Qualidade (7 pares × 3 revisores, fala real) e comparação `e4b` × nemotron — `tools/quality_matrix.py`; resultado em [relatorio-avaliacao.md](relatorio-avaliacao.md)
- [x] **8.** Abrir a interface no navegador (motores, contexto, dispositivo, whisper.cpp) e rodar a build do executável *(OK; build só em venv limpo, ver modelos.md 6.5)*

## 1. Validar os motores Whisper (poucos minutos)

```
pip install -r requirements-engines.txt          # só o que for testar (ver o arquivo)
python tools/validate_engines.py D:/clips/fala_curta.mp3 --language pt-BR --model tiny
```

Para cada motor instalado, baixa o modelo `tiny`, gera a legenda e mostra tempos, as primeiras legendas e avisos. Confirma que as chamadas às bibliotecas reais funcionam (os testes usam dublês). Use um áudio curto (10–30 s) com fala clara. Para testar a diarização: defina `HF_TOKEN` e passe `--diarize` (só o WhisperX).

Teste cada dispositivo que a máquina tiver, fixando-o: `MT_DEVICE=cpu`, `MT_DEVICE=cuda`, `MT_DEVICE=xpu` ou `MT_DEVICE=mps` antes do comando (ou o campo *Dispositivo* em Configurações). O faster-whisper só aceita CPU e CUDA.

Pontos que merecem atenção na primeira execução: API do WhisperX (alinhamento e diarização mudam entre versões), `torch`/CUDA ou CPU, e se o modelo de alinhamento do idioma baixou.

## 2. Medir velocidade e qualidade

Crie um manifesto (lista JSON) com clipes e, se tiver, a referência:

```
[
  {"audio": "D:/clips/aula.mp3", "language": "pt-BR", "reference": "D:/clips/aula.txt", "reference_srt": "D:/clips/aula.srt"}
]
```

- `reference`: texto corrido correto, **no idioma de saída** (habilita o **WER**).
- `reference_srt`: legenda correta (habilita o **erro de tempo**).
- Ambos são opcionais; sem eles ainda saem o tempo por etapa e as violações das regras de legenda.

```
python tools/evaluate.py manifesto.json --engine ollama --model gemma4:e4b --parallel 1 2 4 --translate-batch 1 8 --out resultado.json
```

Cada combinação gera uma linha: tempo total, **RTF** (tempo de processamento ÷ duração do áudio; menor é melhor; 0,5 = o dobro da velocidade real), tempo por etapa (modelo, extração, VAD, transcrição, tradução, montagem), **WER**, quantidade de legendas que violam cada regra e erro de tempo (mediana e p90 em segundos).

### Comparar os motores Whisper entre si
Use os mesmos clipes e o mesmo tamanho de modelo (ex.: turbo) em cada motor:

```
python tools/evaluate.py manifesto.json --engine openai-whisper --model large-v3-turbo --out ow.json
python tools/evaluate.py manifesto.json --engine faster-whisper  --model large-v3-turbo --out fw.json
python tools/evaluate.py manifesto.json --engine whisperx        --model large-v3-turbo --out wx.json
python tools/evaluate.py manifesto.json --engine whisper-cpp     --model large-v3-turbo-q5_0 --out wc.json
python tools/evaluate.py manifesto.json --engine whisper-cpp --model large-v3-turbo-q5_0 --out wcg.json   # usa a GPU NVIDIA se houver (dispositivo 'cpu' nas Configurações força a CPU)
```
O **whisper.cpp** (`whisper-cpp`) é um motor só, com backend automático (GPU NVIDIA/CUDA se houver, senão CPU); o app baixa o executável do backend escolhido ao usá-lo (ou use Configurações / `WHISPER_CPP_PATH` / `WHISPER_CPP_CUDA_PATH`) e usa modelos GGML (o quantizado `q5_0` é a variação leve). Compare RTF e WER; a pergunta é se um motor sem PyTorch fica perto dos demais em qualidade e velocidade na **sua** GPU (o whisper.cpp depende do backend com que o executável foi compilado).

### O que decidir com os números
| Pergunta | Compare |
|----------|---------|
| Vale o paralelismo? Qual valor? | `--parallel 1 2 4` (RTF e uso de VRAM) |
| A tradução em lote ajuda? | `--translate-batch 1 8` (RTF e WER) |
| O VAD ajuda? | com e sem `--no-vad` |
| Tamanho do trecho | `--chunk 8`, `12`, `20` |
| Contexto/glossário melhora? | com e sem `--context "termos…"` |
| Qual motor/modelo? | `--engine ollama|openai-whisper|faster-whisper|whisperx` com `--model` |
| Alinhamento forçado compensa? | erro de tempo do WhisperX × Ollama |

Em cada rodada use os **mesmos clipes** e inclua PT-BR e EN, fala limpa e com ruído de fundo (o limiar do VAD foi testado só com sinais sintéticos). Lembrete do histórico do projeto: um clipe só engana; use vários.

## 3. O app mostra os tempos

Ao terminar, a mensagem do trabalho traz o resumo (ex.: `42 s para 600 s de áudio (RTF 0.07: 14.3x o tempo real)`) e a API devolve `timings` por etapa em `/api/jobs/<id>`.

## 4. Sonda da opção "Qualidade" (pequeno teste, poucos minutos de GPU)

Objetivo: decidir se vale construir o pipeline multiagente da opção **Qualidade** (ver seção 11 de [otimizacoes-futuras.md](otimizacoes-futuras.md)) com dados antes de escrever agentes. Dois testes curtos, em ambientes separados.

### 4.1 Divergência entre duas gerações do Ollama (`tools/quality_probe.py`)

```
python tools/quality_probe.py D:/clips/aula.mp3 --language pt-BR --model-a gemma4:e4b --model-b gemma4:e2b --reference D:/clips/aula.txt --keep-fragments D:/clips/frag --out sonda.json
```

Transcreve os **mesmos fragmentos** com dois modelos de áudio, **em fases** (todos no A, depois todos no B: GPU única), limpa o texto e mede, por fragmento, quanto A e B discordam (`metrics.divergence`). Com `--reference`, informa o WER de cada modelo e uma aproximação do WER dos fragmentos que concordam versus os que divergem. `--keep-fragments` guarda os fragmentos em WAV com um `manifesto.json` (início, fim, texto de A e B, se divergiu), que é o formato previsto para a pasta de trabalho do pipeline.

**Como ler**
| Resultado | Conclusão |
|-----------|-----------|
| Poucos fragmentos divergem (< ~30 %) **e** os que divergem têm WER maior | O portão de divergência funciona: revisar só esses fragmentos é barato e útil |
| A maioria diverge | Os modelos são diferentes demais (ou há ruído): ajustar o limiar (`--threshold`) ou trocar o par de modelos |
| Quase nada diverge | Os modelos erram juntos: o par não agrega; trocar um deles (ou incluir o Whisper como 3ª geração) |
| Divergentes e concordantes têm WER parecido | A divergência não denuncia erro neste áudio: a ideia perde força |

Repetir com 2–3 clipes (PT-BR e EN, fala limpa e com ruído) e com pares diferentes (`gemma4:e4b` × `nemotron-3.5-bolt:e4b-q4_K_M`, × `lfm3-omni:2b-q4_K_M`).

### 4.2 Compatibilidade do Google ADK com Ollama e áudio (`tools/adk_probe.py`)

Num **ambiente virtual separado** (o ADK traz muitas dependências; não instalar no ambiente do app):

```
python -m venv .venv-adk
.venv-adk\Scripts\activate
pip install google-adk litellm requests numpy
python tools/adk_probe.py D:/clips/frag/0000.wav --model gemma4:e4b --expect "palavra que aparece no áudio"
```

| Teste | Pergunta | Se falhar |
|-------|----------|-----------|
| T1 texto simples | O agente fala com o Ollama via `LiteLlm("ollama_chat/<modelo>")`? | Revisar a configuração do LiteLLM/`OLLAMA_API_BASE` |
| T2 tool calling | O modelo local chama uma ferramenta pelo ADK? | Usar só agentes de fluxo (`SequentialAgent`/`ParallelAgent`/`LoopAgent`) trocando texto por `output_key` |
| T3 ferramenta de áudio | O agente chama `ouvir_fragmento` (nosso `OllamaClient`) e usa o texto? **Caminho previsto para o revisor** | Sem T3 e sem T2, o revisor vira etapa do nosso código (sem tool calling), com o ADK só para a orquestração |
| T4 áudio nativo | A mensagem com `inline_data` de áudio chega ao modelo? | Esperado falhar (há relatos de áudio descartado/recusado no caminho LiteLLM); se passar, melhor |

O script foi escrito de memória sobre a API do ADK (`LlmAgent`, `LiteLlm`, `Runner`, `InMemorySessionService`): ajuste se a versão instalada diferir. Saída: OK/FALHOU por teste, com o motivo.

### 4.3 Decisão depois da sonda
1. 4.1 favorável **e** T3 (ou T2) OK → construir a opção Qualidade como planejado (seção 11 do roadmap).
2. 4.1 favorável, ADK problemático → mesma arquitetura, com o nosso código orquestrando (ou agentes de fluxo do ADK sem tool calling).
3. 4.1 desfavorável → não construir; investir em motores/regras de legenda.

## 5. Matriz da opção Qualidade (`tools/quality_matrix.py`)

```
python tools/quality_matrix.py real.json holdout.json --cache cache/ --out resultado/ --reviewers gemma4:e4b nemotron-3.5-bolt:e4b-q4_K_M gemma4:12b --compare gemma4:e4b nemotron-3.5-bolt:e4b-q4_K_M
```

Roda cada gerador (`e2b`, `e4b`, Whisper `medium`, `large-v3`, `turbo`) **uma vez por clipe** (cache em disco) e, para cada um dos 7 pares e cada revisor, refaz só o portão, a revisão e o juiz.
Antes de cada geração com Whisper o script descarrega os modelos do Ollama (a VRAM da XPU não comporta os dois). `--compare` faz a comparação direta de dois modelos (fragmentos idênticos, vitórias de cada um, teste dos sinais).
Saída: `matriz.json`. Os resultados de 2026-11-11 estão em [dados/matriz-qualidade-2026-11-11.json](dados/matriz-qualidade-2026-11-11.json).

## 6. Modelos candidatos para avaliações futuras (registro de 2026-11-11)

Qwen3-Omni / ASR, Nemotron 3.5 Transcribe (incluindo o `30B-A4B-NVFP4` e as versões GGUF do NeMo-Speech.cpp, `Q2_0_M` e `Q2_0_L`),
`nemotron-3.5-asr-streaming-0.6b`, Voxtral Small / Mini e Cohere Transcribe. **Não foram testados**; a lista, o que verificar antes de
baixar e o método de avaliação estão em [otimizacoes-futuras.md](otimizacoes-futuras.md) seção 12.


## 7. Laço de refinamento e modos de qualidade (feito em 2026-11-11)

`tools/quality_loop_experiment.py` mede geradores × revisores com cache de chamadas (v1, v2 e o autorrefinamento sem WhisperX com `--pairs`); `tools/experimento_filme/` prepara o trecho de filme (corte exato, legenda original, referência traduzida) e roda o juiz cego de fidelidade. Resultados e como reproduzir: [experimento-loop-refinamento.md](experimento-loop-refinamento.md).
