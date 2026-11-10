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
