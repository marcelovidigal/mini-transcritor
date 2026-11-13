# Otimizações futuras (roadmap técnico)

> Itens **não implementados**, ordenados por valor/esforço. A base de evidências está em [pre-analise-asr.md](pre-analise-asr.md) (NeMo/Nemotron e Whisper) e [modelos.md](modelos.md) (testes de qualidade com o Ollama). Data da análise: 2026-11-08.

## 1. Timestamps reais (prioridade alta)

**Problema:** os modelos de áudio do Ollama não retornam tempos; hoje as legendas usam o tempo do trecho (~12 s) distribuído por proporção de texto (erro de alguns segundos).

**Diretriz decidida:** ao introduzir Whisper, **dar preferência às variantes que entregam timestamps** (por palavra ou por segmento), nesta ordem de preferência:

| Prioridade | Variante | Por que | Modelos preferidos |
|:--:|----------|---------|---------------------|
| 1 | **WhisperX** (faster-whisper + alinhamento wav2vec2) | tempo **por palavra** preciso, saída **SRT** documentada, diarização opcional, há modelo de alinhamento para **português** | `large-v3-turbo` (rápido), `large-v3` (máx. qualidade), `medium` (menor VRAM) |
| 2 | **faster-whisper** (CTranslate2) | `word_timestamps=True`, VAD Silero, não precisa de ffmpeg, int8 em CPU/GPU, ~4× mais rápido que o original | `large-v3-turbo`, `large-v3`, `small`/`medium` para CPU |
| 3 | **whisper.cpp** (GGML) | binário sem Python, **Windows/Linux/macOS**, CPU/CUDA/Vulkan/Metal; timestamps por palavra ainda **experimentais** (`-ml 1`) | `large-v3-turbo`, `medium`, `small` (quantizados q5_0) |
| — | Whisper original (`openai/whisper`) | referência; timestamps nativos por segmento podem errar "vários segundos" | só como baseline |
| — | `distil-whisper` | **só inglês** | descartado para português |

Observações: o `turbo` **não** serve para tradução (usar `medium`/`large` se houver tradução nativa para inglês); a tradução para outros idiomas continua sendo a 2ª etapa por texto já existente. Janela de 30 s ⇒ usar VAD e trechos com sobreposição.

**Alternativa sem trocar o backend (híbrida):** manter a transcrição pelo Ollama e usar apenas o **alinhador** (WhisperX/CTC) para obter os tempos de cada palavra. Preserva o fluxo atual e melhora só o ponto fraco (tempo).

**Fora do Whisper (com timestamps):** `nvidia/parakeet-tdt-0.6b-v3` (NeMo; 0,6 B, português incluso, tempo por palavra/segmento/caractere, CC-BY-4.0), `canary-1b-v2` e, para hardware de datacenter, `Nemotron-3.5-Transcribe-30B-A4B` (tempo por palavra e diarização). Melhor encaixe: **modo servidor/Docker em Linux**.

**Critério de aceite sugerido:** SRT com erro de tempo mediano < 0,3 s; WER em português ≤ o do `gemma4:e4b` nos clipes de [modelos.md](modelos.md) e em áudios reais do usuário.

## 2. Interface de backend

> **Status (2026-11-10):** feito para OpenAI Whisper, faster-whisper e WhisperX (`transcritor/engines.py`, catálogo `whisper_models.txt`), com o Ollama como motor padrão. O whisper.cpp entrou como motor `whisper-cpp` (via executável `whisper-cli`; backend CPU ou CUDA automático). Falta a validação com os pacotes e o executável reais. Ver seção 8.

Hoje `pipeline.py` fala direto com `OllamaClient`. Extrair uma interface (`transcribe(audio) -> segmentos com tempos`) com implementações `OllamaBackend` (atual), `WhisperCppBackend`, `FasterWhisperBackend`. Pré-requisito para o item 1. O catálogo `audio_models.txt` ganharia a coluna "backend".

## 3. Instalação automática de backends

Seguir o padrão do FFmpeg/Ollama: detectar → configurar caminho → baixar localmente no primeiro uso (whisper.cpp: binário + modelo GGML em `whisper/` ao lado do app; modelo `ggml-large-v3-turbo` quantizado).

## 4. Qualidade e robustez (independente do backend)

- **VAD** (Silero) para cortar silêncios e reduzir alucinação; trechos com **sobreposição** e reconciliação do texto repetido.
- **Pós-processamento de legenda:** limite de caracteres por linha/duração, quebra por pontuação usando os tempos reais.
- **Detecção de idioma** automática e exibição ao usuário; contexto (prompt) com nomes próprios/glossário.
- **Avaliação contínua:** transformar o teste de WER (clipes sintéticos + áudios reais, com falantes e ruído) num script do repositório (`tools/eval_models.py`) e registrar resultados em [modelos.md](modelos.md).
- Revisitar a regra de recusa/alucinação do `gemma4:12b` ("I cannot provide a transcription") com um prompt de reforço e *retry*.

## 5. Modo servidor (Fase 2)

Fila persistente (Redis/RQ), armazenamento de objetos, quotas por usuário, múltiplos workers por GPU, métricas e limite de requisições por IP (ver [servidor.md](servidor.md) §9).

## 6. Distribuição

Assinatura de código (Windows) e assinatura + notarização (macOS); instalador (`.msi`/`.dmg`); download automático do FFmpeg em Linux/macOS; build macOS Intel; imagem Docker publicada num registro.

## 7. Modelos a reavaliar periodicamente

- Se o Ollama passar a suportar **Qwen2.5-Omni / Qwen3-Omni / Qwen2-Audio**, **Voxtral**, **Phi-4 Multimodal** ou **SALMONN** (hoje ausentes da biblioteca oficial), incluir em `audio_models.txt` com a tag descritiva e testar.
- Nemotron no Ollama: só o `nemotron-3.5-bolt:e4b-q4_K_M` processa áudio (empata com o `gemma4:e4b`); o `nemotron3` (33B) **não** aceita áudio no Ollama. Reavaliar o `bolt:74b-a9b` (25 GB, IQ2, tem projetor, **não testado**) e se o `nemotron3` ganhar projetor em novas tags. Detalhes em [modelos.md](modelos.md).

## 8. Motores Whisper no executável (decisão pendente)

**Situação atual.** Os motores OpenAI Whisper, faster-whisper e WhisperX são pacotes Python opcionais (`requirements-engines.txt`) e só funcionam quando o app roda **a partir do código-fonte**. O executável (PyInstaller) é gerado sem eles e continua leve (~64 MB, só Ollama): no executável os três aparecem no seletor como "não instalado".

**Por quê.** O **OpenAI Whisper** e o **WhisperX** dependem do **PyTorch** (alguns GB; mais com CUDA); o WhisperX ainda traz o faster-whisper e modelos extras (alinhamento e diarização). O **faster-whisper é independente e não usa PyTorch** (usa CTranslate2; na GPU precisa só das bibliotecas CUDA), por isso é o único candidato realista a entrar num executável leve — uma quarta opção a avaliar: embutir **somente o faster-whisper**, sem diarização. Embuti-los inflaria a pasta de release para vários GB, deixaria a build lenta e frágil (PyTorch com CUDA dentro do PyInstaller costuma dar problemas) e oneraria quem só usa o Ollama.

**Opções avaliadas**

| Opção | Prós | Contras |
|-------|------|---------|
| A. Manter assim (atual) | Executável leve e simples; menos risco | Quem quer Whisper precisa rodar do código-fonte |
| B. Segundo executável "completo" (release separado, com os motores) | Funciona sem Python; o usuário escolhe o pacote | Vários GB; build por plataforma e por tipo de GPU (CUDA/CPU); manutenção dobrada |
| C. O executável instala os pacotes sozinho numa pasta própria ao selecionar um motor (como o FFmpeg) | Experiência mais amigável; executável continua leve | Mais complexo e arriscado (`pip` e compatibilidade com o Python embutido, versão do CUDA, downloads de GB); precisa de um ambiente isolado |

**Recomendação.** Ficar com A por enquanto. Só avaliar B ou C depois de um teste real com os pacotes (um modelo pequeno, como `tiny` ou `base`, em cada motor) confirmar que os motores funcionam bem e que o ganho de qualidade/velocidade justifica o peso. Se for investir, C tende a atender melhor o usuário final; B é mais previsível para distribuir.

**Pré-requisitos comuns a B e C:** validar as APIs reais (principalmente WhisperX: alinhamento e diarização), medir o WER dos motores no conjunto de referência e decidir o suporte a GPU (CUDA) por plataforma.

## 9. GPU Intel (XPU / OpenVINO)

A máquina de desenvolvimento tem uma **Intel Arc Pro B60**. O PyTorch instalado no Python global (2.9.1+cu130) é uma build **CUDA**: `torch.xpu.is_available()` e `torch.cuda.is_available()` dão `False`, então OpenAI Whisper e WhisperX rodam na **CPU**; o faster-whisper (CTranslate2) também não usa GPU Intel.

- **PyTorch XPU (preparado):** o app já escolhe `cpu`, `cuda`, `xpu` ou `mps` (`engines.pick_device`; campo *Dispositivo* em Configurações; `MT_DEVICE`). Ao instalar a build `+xpu` do PyTorch (num ambiente virtual separado), o modo automático passa a usar a GPU Intel no OpenAI Whisper e no alinhamento/diarização do WhisperX. **Falta validar em hardware real** com `tools/validate_engines.py` (operadores do Whisper em XPU/MPS nem sempre são suportados; nesse caso o app recai na CPU no WhisperX e falha com mensagem clara no OpenAI Whisper).
- **OpenVINO:** `openvino` e `onnxruntime-openvino` já estão instalados; Whisper em formato OpenVINO (`optimum-intel` / `openvino-genai`) seria um motor novo para a GPU Intel.
- O **Ollama** já usa a GPU Intel e segue como caminho principal nesta máquina.

## 10. Diarização independente de motor (a avaliar; o WhisperX permanece)

**Decisão (2026-11-10):** o WhisperX e o PyTorch **permanecem**. A diarização do app hoje existe só no WhisperX (pyannote, em PyTorch). Antes de qualquer remoção, avaliar uma alternativa sem PyTorch.

**Ideia.** Diarização ("quem fala quando") é uma etapa separada da transcrição: um módulo `diarization.py` rodaria sobre o áudio e associaria cada segmento ao falante com maior sobreposição de tempo, para qualquer motor com timestamps reais (OpenAI Whisper, faster-whisper, whisper.cpp). Candidato: modelos em ONNX via `sherpa-onnx` (segmentação + embedding de voz) no `onnxruntime`, que também poderia usar a GPU Intel pelo `onnxruntime-openvino` já instalado. Pontos a confirmar antes: modelos exatos, licenças e acesso (o pyannote exige token e aceite de licença no Hugging Face).

**Riscos.** O pyannote 3.1 é a referência de qualidade; alternativas em ONNX tendem a errar mais na contagem de falantes e em falas sobrepostas. Sem timestamps por palavra (o alinhamento do WhisperX), um segmento com troca de falante no meio recebe um só rótulo.

**O que o WhisperX faz além de diarizar** (e que também teria de ser substituído para dispensá-lo): transcrição em lote com VAD (o faster-whisper já oferece) e alinhamento por palavra com wav2vec2, em PyTorch (o faster-whisper tem `word_timestamps`, menos preciso).

**Plano de avaliação.** (1) Medir os motores atuais e o WhisperX com um áudio de várias vozes (`tools/evaluate.py`); (2) prototipar o módulo ONNX e comparar a taxa de erro de diarização e o tempo com o pyannote; (3) só então decidir. Se ficar perto, o PyTorch passa a ser necessário apenas para o OpenAI Whisper original, que o faster-whisper e o whisper.cpp substituem.

## 11. Opção "Qualidade" (pipeline multiagente com Google ADK) — a avaliar

> Documento completo, com diagramas: [proposta-multiagente.md](proposta-multiagente.md).

**Decisão (2026-11-10):** além de escolher motores, o usuário poderá escolher a opção **Qualidade**, cuja arquitetura e modelos são **fixos** no pipeline. O foco é o **Ollama**. Dependência prevista: Google ADK (num ambiente separado; fora do executável leve). Antes de construir, a sonda da seção 4 de [avaliacao.md](avaliacao.md) valida a ideia.

**Desenho (padrão composto do guia de padrões multiagente do ADK):**

```
Preparo (código)        extrai áudio, VAD, fragmentos em work/<id>/ + manifesto
Geração (padrão 3)      G1: modelo de áudio A · G2: modelo de áudio B (Ollama) · G3: Whisper (opcional, depois)
Portão de divergência   código determinístico: concordam → aceita; divergem/suspeito → revisão
Revisor (padrão 5)      modelo de áudio REOUVE o fragmento (ferramenta ouvir_fragmento) → JSON com veredito e evidência
Juiz (finalizador)      aplica o veredito com travas (mudança pequena e justificada); na dúvida, geração prioritária
Especialistas (pad. 1)  termos/glossário → pontuação e segmentação → revisão da tradução
Regras Netflix          e relatório de qualidade (por fragmento) salvo na pasta de trabalho e resumido no trabalho
```

Nomes: o agente final é o **juiz/árbitro** (*LLM-as-a-Judge*) ou **agregador** (*Mixture-of-Agents*); combinar hipóteses de várias fontes é **ROVER** (votação) ou **correção generativa de erros** (um LLM recebe as N hipóteses).

**Regras:**
- **Fragmentos de áudio:** guardados em `work/<id>/` e removidos **somente quando o trabalho conclui ou dá erro** (decisão do usuário), nunca durante a revisão. Sugestão em aberto: apagar também ao cancelar, com opção para manter (útil na análise). ~2 MB por minuto de áudio.
- **GPU única:** a geração roda **em fases** (todos os fragmentos num modelo, depois no outro); um `ParallelAgent` com modelos diferentes não paraleliza de verdade.
- **Anti-regressão:** o revisor devolve JSON estruturado; o juiz só aceita mudanças pequenas e com evidência. O risco principal é "corrigir" o que estava certo.
- **ADK e áudio:** o caminho padrão (`LiteLlm` + `ollama_chat/`) tem relatos de descartar áudio; o plano é uma **ferramenta própria** que usa o nosso `OllamaClient`. Tool calling com modelos locais precisa ser verificado (sonda T2/T3). O ADK entra nas etapas com papéis (revisor, juiz, especialistas); a geração pesada segue no nosso pipeline.
- **Custo:** dobra o tempo base: por isso é uma opção separada da rápida.

**Ordem:** (1) pasta de trabalho + manifesto; (2) portão de divergência + relatório; (3) sonda ADK; (4) revisor, juiz e especialistas, cada um medido com `tools/evaluate.py` (WER antes/depois e tempo extra); (5) Whisper como 3ª geração se os números justificarem.

**Resultado da sonda (2026-11-11), seção 11:** no material sintético PT-BR (1103 palavras), o `gemma4:e4b` já erra ~2 %; um revisor perfeito reduziria para 1,6–1,9 % e o portão de divergência (par `e4b` × `e2b`/`bolt`) só marca 10–20 % dos fragmentos com erro, porque os modelos erram juntos. O ADK funciona com o Ollama (chamada de ferramenta e ferramenta de áudio própria OK; áudio nativo via LiteLLM falha). **Decisão sugerida:** não construir a opção Qualidade completa agora; repetir a sonda com **fala real** (alguns minutos de gravação com transcrição conferida) antes de investir. Ver [modelos.md](modelos.md) seção 6.3.

**Resultado final da avaliação (2026-11-11), seção 11:** a matriz completa (7 pares de geradores × 3 revisores, fala real) **confirmou a opção Qualidade**: `e4b` + Whisper `turbo` + revisor `gemma4:12b` reduz o WER de 4,85 % (Whisper sozinho)
para 3,99 %. O protótipo está implementado (`transcritor/quality_engine.py`) com essa composição como padrão. Próximos passos: validar com fala espontânea e outros idiomas, especialistas (glossário, pontuação, revisão da tradução)
e a camada do ADK sobre o `OllamaClient` (sem LiteLLM). Relatório completo: [relatorio-avaliacao.md](relatorio-avaliacao.md).

## 12. Modelos de transcrição a avaliar no futuro (registrado em 2026-11-11)

Lista passada pelo usuário para **avaliações futuras**; nada abaixo foi baixado, testado ou verificado ainda (nomes, tamanhos e
capacidades são os informados e precisam ser conferidos na fonte antes de qualquer download: ver a regra do manifesto/projetor em
[modelos.md](modelos.md) seção 4). Nada disto entra nas opções do app antes de passar pela mesma bateria dos demais.

| Candidato | O que foi informado | Pontos a verificar |
|-----------|---------------------|--------------------|
| **Qwen3-Omni / ASR** | modelo multimodal da família Qwen com variante de reconhecimento de fala | existe tag no Ollama com projetor de áudio? tamanho; idiomas; formato de entrada |
| **Nemotron 3.5 Transcribe** | modelo da NVIDIA voltado à transcrição | confirmar se é o mesmo que o `...-30B-A4B-NVFP4` abaixo; disponibilidade no Ollama |
| **`NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4`** — **cliente vLLM implementado** (`nemotron-vllm`; não executável nesta máquina: exige Blackwell) | 30B (4B ativos), formato NVFP4, focado em **arquivos gravados de até 1 hora** | NVFP4 é formato para GPU NVIDIA: roda na Intel Arc? limite de 1 h × nossos fragmentos de ~12 s |
| **Versões GGUF (NeMo-Speech.cpp)** — **IMPLEMENTADO em 2026-11-12** (motor `nemo-speech-cpp`, ver [nemotron-nemo-speech.md](nemotron-nemo-speech.md)) | compilações `Q2_0_M` (13,6 GB) e `Q2_0_L` (15,5 GB), mais 1,6 GB do encoder; para GPU de consumo de 16 GB ou Apple Silicon com 24 GB | motor próprio (`NeMo-Speech.cpp`), não é o Ollama: exigiria um novo motor opcional como o whisper.cpp; quantização Q2 agressiva (ver o caso do IQ2 em [modelos.md](modelos.md)) |
| **`nemotron-3.5-asr-streaming-0.6b`** — **IMPLEMENTADO** como opção do `nemo-speech-cpp` (WER 7,24 % no clipe de teste) | 600 M de parâmetros, ultraleve, **streaming/tempo real** | útil só se o app ganhar um modo ao vivo; como gerador B rápido, comparar com o WhisperX `turbo` |
| **Voxtral Small / Mini** | família de áudio da Mistral | tamanhos; suporte no Ollama; idiomas (pt-BR) |
| **Cohere Transcribe** | modelo de transcrição da Cohere | se é local ou só API (API em nuvem foge do foco local do projeto) |

**Como avaliar (quando chegar a hora), sem mudar o método:** (1) conferir fonte, licença, tamanho e a camada de projetor/áudio **antes**
de baixar; (2) tag explícita e descritiva, nunca `latest`; (3) FLEURS (6 clipes) como triagem e o trecho de filme de 45 min
(`docs/experimento-loop-refinamento.md`) como teste principal; (4) papéis possíveis: gerador sozinho, **gerador B no lugar do WhisperX**
ou, se for um modelo de linguagem com áudio, **revisor**; modelos só de ASR (como os Whisper) são geradores, nunca revisores;
(5) medir WER, fidelidade pt-BR, tempo e memória no mesmo harness (`tools/quality_loop_experiment.py`).
