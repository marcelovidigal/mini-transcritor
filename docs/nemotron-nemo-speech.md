# Nemotron da NVIDIA no app: NeMo-Speech.cpp e servidor vLLM

Registro de 2026-11-12. Dois motores novos, **opções individuais** (não entram nos modos de Qualidade por enquanto):

| Motor (id) | Modelos | Como roda |
|---|---|---|
| **NeMo-Speech.cpp** (`nemo-speech-cpp`) | `nemotron-3.5-transcribe-30b-a4b-q2_0_m` (13,6 GB + codificador 1,7 GB), `nemotron-3.5-transcribe-30b-a4b-q2_0_l` (15,5 GB + 1,7 GB), `nemotron-3.5-asr-streaming-0.6b` (0,74 GB) | executável C++ oficial da NVIDIA (ggml), sem Python/PyTorch |
| **Nemotron Transcribe NVFP4** (`nemotron-vllm`) | `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4` | o app é **cliente** de um servidor vLLM seu (GPU NVIDIA Blackwell) |

## Correção sobre o NVFP4 ("via nemo_asr")

A pergunta era se o NVFP4 roda pela biblioteca `nemo_asr`. **Não é o caminho documentado.** O cartão do modelo no Hugging Face diz que o NVFP4 é para **GPU NVIDIA Blackwell** e é servido pelo **vLLM com o plugin do NeMo Speech** (`VLLM_PLUGINS=nemo_speechlm`, `vllm serve ...`); o caminho "em processo" do cartão é o do Hugging Face Transformers (`AutoModelForMultimodalLM`), e é para o checkpoint BF16. Por isso o motor é um cliente da API OpenAI-compatível do vLLM (`/v1/chat/completions` com `audio_url`), configurado em *Configurações → Servidor vLLM*. Nesta máquina (Intel Arc Pro B60) ele **não pode ser executado**: foi testado só com um servidor simulado nos testes automáticos.

## NeMo-Speech.cpp

- **Binário:** release oficial (`github.com/NVIDIA/NeMo-Speech.cpp`, última versão estável), baixado ao gerar a legenda para `<app>/nemocpp/` com SHA-256 conferido (`transcritor/nemocpp_manager.py`). O pacote é escolhido pela máquina: **CUDA** (GPU NVIDIA), **Vulkan** (qualquer outra GPU: Intel Arc, AMD), **Metal** (Apple Silicon) ou CPU. Nesta máquina: Vulkan (22 MB), que enxerga a Arc Pro B60 (23,8 GiB).
- **Modelos:** arquivos GGUF baixados para `<pasta de dados>/nemocpp-models/<modelo>/` (retomáveis, tamanho e SHA-256 fixos em `engines.NEMO_MODELS`, URLs com revisão fixa). O Transcribe-30B é um **par** (decodificador + codificador `Encoder-F16`). O 0,6 B vem do repositório da NVIDIA; as conversões do 30B (Q2_0_M/L) são da **AtomicChat** (comunidade, não da NVIDIA; a licença é a OpenMDW-1.1 do modelo original).
- **Transcribe-30B:** cada trecho de fala do VAD vira um arquivo e **uma única execução** do CLI processa a pasta (`--concurrency 4`, modelo carregado uma vez). Progresso real (`[k/n]`), cancelamento mata o subprocesso. O idioma não vem do modelo: é detectado pelo texto, para não traduzir sem necessidade. O contexto/glossário entra como instrução (`--instruction`).
- **ASR Streaming 0,6 B:** janelas de até 1 h do áudio, `--language auto`, tempos por palavra do modelo.
- **Por que não uma janela longa no 30B:** os tempos por palavra (cabeçote CTC) saem **empacotados** (ex.: "setecentos" em 91,76 s e "sessenta" em 110,82 s) e o filtro de alucinação descartava os trechos; além disso era 3,4x mais lento (305 s contra 89 s). Com trechos do VAD o tempo vem do VAD; tempos por palavra só são usados se forem plausíveis (ordem, limites, no máximo 20 caracteres/s).
- **Limite do Windows (MAX_PATH):** `nemo-speech pull` falha com "cannot write partial-download revision marker" quando a pasta de modelos tem caminho longo. O app não usa o `pull`: baixa os arquivos ele mesmo e passa o caminho local ao `--model`.

## Resultado medido (1 clipe de 273 s, fala real em PT-BR, Arc Pro B60 via Vulkan)

| Modelo | WER | Tempo de transcrição | RTF |
|---|---|---|---|
| Transcribe-30B Q2_0_M | **4,52 %** | 48 s | 0,17 |
| ASR Streaming 0,6 B | 7,24 % | 22 s | 0,08 |

Um único clipe: é indicação, não conclusão. O Q2_0_L não foi executado (download de 15,5 GB; URL, tamanho e SHA-256 foram conferidos). A comparação com WhisperX e Ollama no mesmo conjunto (filme + FLEURS) fica para a próxima avaliação.

## Diferenças de comportamento

- Música/ruído: o 30B pode inventar texto nesses trechos; como só vão os trechos de fala do VAD, o risco é menor.
- O 0,6 B ignora o contexto/glossário (o app registra um aviso no log).
- `nemotron-vllm` só aparece "disponível" se o servidor responder em `http://127.0.0.1:8000/v1` (ou no endereço configurado).

## GitHub e release

`whispercpp/` e `nemocpp/` ficam no `.gitignore`: são **cache de executáveis baixados** (1,2 GB e 74 MB aqui), não código. O que vai para o repositório é o código que os baixa (`whispercpp_manager.py`, `nemocpp_manager.py`, `engines.py`). O CI e o release instalam só `requirements.txt` e o PyInstaller não empacota esses executáveis nem os motores Python; ao usar o motor, o app baixa o executável do release oficial do projeto de origem (SHA-256 conferido) para a pasta do app, e os modelos para a pasta de dados. No modo servidor nada é baixado (ver `docs/servidor.md`).
