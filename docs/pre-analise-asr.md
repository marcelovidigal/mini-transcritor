# Pré-análise: NVIDIA NeMo ASR (Nemotron) e família Whisper

> **Status:** pesquisa preliminar para decidir uma possível evolução do backend. **Nada aqui está implementado**: o Mini Transcritor continua usando apenas o **Ollama** (ver [prd.md](prd.md)). Dados coletados em 2026-11-08 em fontes primárias (cartões de modelo no Hugging Face, READMEs no GitHub, páginas do Ollama); itens não confirmados estão marcados **[não verificado]**. Números de WER vêm dos próprios autores, em benchmarks diferentes, e **não são comparáveis entre si** nem com os testes deste projeto ([modelos.md](modelos.md)).

> **Decisão registrada:** ao introduzir Whisper, preferir as variantes **com timestamps** (WhisperX > faster-whisper > whisper.cpp). Plano e critérios: [otimizacoes-futuras.md](otimizacoes-futuras.md).

## 1. Por que olhar para fora do Ollama

O pipeline atual usa modelos **multimodais** (LLMs que "ouvem") via Ollama. Isso tem duas limitações já medidas neste projeto:

| Limitação atual | Efeito |
|-----------------|--------|
| Os modelos de áudio do Ollama não devolvem **timestamps** | As legendas têm tempo aproximado (por trecho de ~12 s), não por palavra |
| Qualidade irregular e alucinações (ex.: `gemma4:12b` recusando transcrever; LFM fraco em português) | Exige revisão manual |

Modelos de **ASR dedicado** (reconhecimento de fala) costumam entregar timestamps por palavra/segmento, longa duração e WER baixo e previsível. Em contrapartida, **não rodam no Ollama** (ver §5) e exigem outro runtime (Python/NeMo, CTranslate2, GGML…).

## 2. NVIDIA NeMo e a família Nemotron/Parakeet/Canary

### 2.1 Sobre o nome "nemotron-3.5-asr"

**Não existe um repositório literalmente chamado `nemotron-3.5-asr`** (pesquisa no Hugging Face, página 1 dos resultados; uma correspondência literal em páginas posteriores não está descartada). Os modelos NVIDIA reais dessa geração são:

| Repositório (Hugging Face) | Tipo | Tamanho | Observação |
|----------------------------|------|---------|-----------|
| `nvidia/nemotron-3.5-asr-streaming-0.6b` | ASR **streaming** multilíngue (FastConformer-RNNT) | 0,6 B | o mais próximo de "nemotron-3.5-asr" |
| `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` | ASR **offline** com LLM de fala (MoE) | 34 B (4,3 B ativos) | variantes `-FP8`, `-NVFP4`, `-DFlash2` (rascunho) |
| `nvidia/nemotron-speech-streaming-en-0.6b` | ASR streaming só em **inglês** | 0,6 B | |
| `nvidia/nemotron-labs-unified-asr-0.6b` | ASR unificado | 0,6 B | listado, **não lido** [não verificado] |

### 2.2 Ficha dos modelos NeMo relevantes

| Modelo | Params | Idiomas / **português** | Licença | Duração máx. de entrada | Timestamps | WER reportado (autor) |
|--------|:------:|--------------------------|---------|-------------------------|-----------|-----------------------|
| `Nemotron-3.5-Transcribe-30B-A4B` | 34 B (4,3 B ativos) | 40 idiomas; **pt-BR e pt-PT** ("Leading") | OpenMDW-1.1 | até 1 hora (16 kHz mono) | palavra (cabeças CTC, opcional) + diarização | Open ASR (9 conjuntos) 4,71 %; multilíngue 4,62 %; longa duração 10,98 %; pt FLEURS ≈ 3 %* |
| `nemotron-3.5-asr-streaming-0.6b` | 0,6 B | 35–40 locales; **pt-BR e pt-PT** | OpenMDW-1.1 | "depende da memória da GPU" | não informado | FLEURS pt ≈ 5,5–6,3 % |
| `nemotron-speech-streaming-en-0.6b` | 0,6 B | só inglês (**sem português**) | NVIDIA Open Model | não informado | não informado no cartão | Open ASR 6,93 % |
| `parakeet-tdt-0.6b-v3` | 0,6 B | 25 idiomas europeus; **pt incluso** (FLEURS 4,76 %, MLS 7,50 %, CoVoST 3,96 %) | CC-BY-4.0 | 24 min (atenção total) ou até 3 h (atenção local) | **palavra, segmento e caractere** | Open ASR 6,34 % |
| `parakeet-tdt-0.6b-v2` | 0,6 B | só inglês | CC-BY-4.0 | 24 min | palavra, segmento, caractere | Open ASR 6,05 % (RTFx 3386) |
| `canary-1b-v2` | ~1 B | 25 idiomas; **pt incluso**; tradução EN↔24 idiomas | CC-BY-4.0 | chunking automático em arquivos longos | palavra e segmento (ASR) | Open ASR 7,15 % (RTFx 749) |

\* valores de resumo extraído da página, não literais.

Observações dos cartões: Parakeet v3 foi treinado com português **europeu** e avaliado em brasileiro; `Transcribe-30B` e o `streaming` listam explicitamente pt-BR.

### 2.3 Como rodar localmente

- **Toolkit NeMo** (Apache-2.0; os modelos têm licenças próprias): `pip install -U nemo_toolkit['asr']`; o README atual pede **Python ≥ 3.12 e PyTorch ≥ 2.7**; GPU NVIDIA/CUDA é "recomendada para inferência" (obrigatória para treino). Há imagem Docker (`nvcr.io/nvidia/nemo-speech`).
- **Alternativas de runtime:** `Transcribe-30B` via **vLLM** (Linux ou WSL2) ou **GGML** (única rota nativa no Windows); modelos 0,6 B também via **NeMo-Speech.cpp (GGUF)**, Transformers ≥ 5.13, e portes da comunidade em ONNX/MLX/CoreML [não verificados].
- **Hardware:** o `Transcribe-30B` mira **1× H100 80 GB** (também listados A100 80 GB, L40S 48 GB em FP8, RTX 5090 em NVFP4, RTX PRO 6000); não há VRAM mínima publicada. Parakeet pede "≥ 2 GB de RAM para carregar" e Canary "≥ 6 GB". Sistema preferencial: **Linux**; suporte nativo ao Windows do NeMo [não verificado].
- **Ollama:** nenhum dos cartões menciona suporte ao Ollama.

### 2.4 Leitura para este projeto

- **`parakeet-tdt-0.6b-v3`**: o candidato mais prático — pequeno (0,6 B), português incluso, **timestamps por palavra**, licença permissiva (CC-BY-4.0). Limite: 24 min por passagem (3 h com atenção local).
- **`Nemotron-3.5-Transcribe-30B-A4B`**: melhor em teoria (WER baixo, 1 h, diarização, pt-BR "Leading"), mas **pede hardware de datacenter** e runtime Linux/vLLM; fora do alcance de PCs comuns.
- **Streaming 0,6 B**: pensado para tempo real; para legenda de arquivos o Parakeet/Canary parecem mais adequados.

## 3. OpenAI Whisper e suas famílias

### 3.1 Modelos do Whisper (README `openai/whisper`, licença MIT)

| Modelo | Params | VRAM | Velocidade relativa* | Observação |
|--------|:------:|:----:|:--------------------:|-----------|
| tiny / tiny.en | 39 M | ~1 GB | ~10× | |
| base / base.en | 74 M | ~1 GB | ~7× | |
| small / small.en | 244 M | ~2 GB | ~4× | |
| medium / medium.en | 769 M | ~5 GB | ~2× | |
| large (v2, v3) | 1550 M | ~10 GB | 1× | v3: 10–20 % menos erros que v2 (geral) |
| **turbo** (large-v3-turbo) | 809 M | ~6 GB | ~8× | large-v3 podado (decodificador de 32 para 4 camadas); **sem treino para tradução** |

\* relativa ao `large`, medida numa A100 com inglês; "pode variar bastante" no uso real. Janela de **30 s**; 99 idiomas, **português incluso** no tokenizador (`pt`). **WER por idioma (inclusive português) não verificado** nas páginas lidas [não verificado]. Placar Open ASR (HF): turbo WER médio 7,83 / RTFx ~200; large-v3 7,44 / ~146. Requer **ffmpeg**.

### 3.2 Derivados para uso local

| Projeto | O que é | Pontos fortes | Pontos de atenção |
|---------|---------|---------------|-------------------|
| **faster-whisper** (SYSTRAN, MIT) | Whisper em CTranslate2 | "até 4× mais rápido, com menos memória"; int8 em CPU/GPU; **timestamps por palavra**; VAD (Silero); **não precisa de ffmpeg**; Python ≥ 3.9. Bench (13 min, large-v2, RTX 3070 Ti): 1 min 03 s (17 s em lote de 8), ~4,5 GB VRAM | GPU exige CUDA 12 + cuDNN 9 (no Windows, DLLs à parte); **SRT não vem pronto** (monta-se a partir dos segmentos) |
| **whisper.cpp** (ggml-org, MIT) | Whisper em C/C++ (GGML), sem Python | **Windows (MSVC/MinGW), macOS (Metal/CoreML), Linux**; CPU, CUDA, Vulkan, ROCm, OpenVINO; modelos em disco: tiny 75 MiB … medium 1,5 GiB, large 2,9 GiB; quantização (q5_0 etc.); VAD | `whisper-cli` lê só **WAV 16 kHz** (converter com ffmpeg); timestamps por palavra **experimentais**; flag de **SRT/VTT [não verificado]** |
| **distil-whisper** (Hugging Face, MIT) | Whisper destilado | ~6× mais rápido, ~49 % menor (`distil-large-v3`: 756 M; WER longa duração 10,8 vs 11,0 do large-v3) | **só inglês**; para multilíngue o README indica o turbo; roda também no faster-whisper |
| **WhisperX** (BSD-2) | faster-whisper + alinhamento + diarização | "70× tempo real" (large-v2, em lote); **timestamps por palavra** via wav2vec2 (há modelo de alinhamento **pt**); **saída SRT** documentada; diarização (pyannote, exige token do Hugging Face) | Diarização "longe de perfeita"; CUDA 12.8; VAD padrão [não verificado] |

Os timestamps nativos do Whisper são descritos pelo WhisperX como "em nível de enunciado, às vezes errados em vários segundos" — daí o valor do alinhamento por palavra.

### 3.3 Whisper no Ollama

A busca do site mostra apenas **uploads da comunidade** (ex.: `mannix/whisper`, descrito como "whisper.cpp large-v3 turbo · 99 idiomas"; outros "copiados do Hugging Face"); **não há modelo oficial** (namespace `library`) nas 20 primeiras respostas, e não foi verificado se esses uploads aceitam áudio pela API do Ollama [não verificado]. Na prática, o Whisper é consumido por **whisper.cpp / faster-whisper**, não pelo Ollama.

## 4. Comparação resumida (para legendas SRT locais)

| Critério | Ollama + multimodal (hoje) | Parakeet-v3 (NeMo) | Whisper turbo / large-v3 (faster-whisper) | whisper.cpp |
|----------|:--:|:--:|:--:|:--:|
| Já integrado ao app | **sim** | não | não | não |
| Timestamps por palavra | não | **sim** | **sim** | experimental |
| Português | depende do modelo (Gemma bom, LFM fraco) | sim (pt incluso) | sim (99 idiomas) | sim |
| Tradução para outro idioma | sim (2ª etapa por texto) | só Canary (EN↔24) | turbo **não**; medium/large sim (para inglês) | idem |
| Duração/arquivo | trechos de 12 s | 24 min–3 h | janela 30 s + VAD | janela 30 s |
| Windows nativo | sim | **[não verificado]** | sim (com DLLs CUDA) | **sim** (sem Python) |
| Hardware | GPU média | GPU NVIDIA (≥ 2 GB p/ 0,6 B) | GPU ~6 GB (turbo) ou CPU int8 | CPU, ou GPU NVIDIA/AMD/Apple |
| Licença | variada | CC-BY-4.0 | MIT | MIT |

## 5. Caminhos possíveis (não decididos)

1. **Manter só o Ollama** (situação atual) e documentar as limitações de tempo e qualidade.
2. **Backend adicional "whisper.cpp"** como binário/servidor local (combina com o empacotamento em pasta e com o download local do FFmpeg; roda sem Python e no Windows). Dá timestamps melhores e português estável.
3. **Backend "faster-whisper"/WhisperX** em Python: melhor qualidade de tempo (alinhamento por palavra), mas traz dependências pesadas (CUDA 12, PyTorch no WhisperX) — mais adequado ao **modo servidor/Docker** que ao executável.
4. **Backend NeMo (Parakeet-v3)**: bom custo-benefício teórico, porém com **Linux como alvo** e toolkit pesado (Python ≥ 3.12, PyTorch ≥ 2.7); mais adequado a Docker no servidor.
5. **Pós-alinhamento híbrido:** transcrever com o modelo do Ollama e usar um alinhador (WhisperX/CTC) só para os tempos — é a opção que preserva o fluxo atual.

Qualquer um dos caminhos 2–5 exige: uma **interface de backend** no código (hoje acoplado a `OllamaClient`), novas dependências/download no primeiro uso e testes de qualidade com áudio real em português. A recomendação desta pré-análise é **validar o caminho 2 (whisper.cpp, `large-v3-turbo`)** com os mesmos clipes do [modelos.md](modelos.md) antes de qualquer implementação.

## 6. Lacunas desta pesquisa

- WER do Whisper **em português** e de qualquer modelo NeMo nos nossos áudios: **não medidos** (só números dos autores, sobre outros conjuntos).
- Suporte do NeMo ao Windows nativo; VRAM mínima de cada modelo; comprimento máximo do `nemotron-3.5-asr-streaming-0.6b` e do `canary-1b-v2`.
- Se existe flag de SRT/VTT no `whisper-cli` e quais são os tamanhos dos modelos quantizados.
- Se os uploads de Whisper da comunidade no Ollama aceitam áudio.
- `nvidia/nemotron-labs-unified-asr-0.6b` e `nvidia/canary-qwen-2.5b` não foram lidos.
