# Arquitetura — Mini Transcritor

## 1. Visão geral

Aplicação monolítica Flask. O navegador conversa com uma API JSON; o processamento pesado roda em **threads de fundo** (um *job* por geração) e o front consulta o progresso por *polling*.

```
Navegador (Bootstrap + JS)
   │  HTTP/JSON  (polling de progresso)
   ▼
Flask (app.py)  ──►  JobManager (threads)
                        │
                        ▼  pipeline.run_job
        ┌───────────────┼───────────────────────────┐
   media.py        audio.py                   ollama_client.py
 (ffprobe: tipo)  (ffmpeg → WAV 16 kHz;   (/api/tags, /api/show,
                   corte em pausas)           /api/chat com áudio)
                        │
                     srt.py  → arquivo .srt
```

## 2. Estrutura de pastas

```
mini-transcritor/
├── app.py                 # fábrica Flask + rotas (modo desenvolvimento)
├── server.py              # entrada do MODO SERVIDOR (waitress, valida a configuração)
├── Dockerfile, docker-compose.yml, Caddyfile, .env.example   # deploy do modo servidor
├── launcher.py            # ponto de entrada do executável (servidor + navegador + encerramento)
├── build.py               # gera dist/MiniTranscritor/ com PyInstaller (+ --archive para release)
├── release.py             # testes → build → smoke test → artefato → (opcional) gh release
├── smoke_test.py          # sobe o executável gerado e confere página/estáticos/API
├── audio_models.txt       # catálogo estático de modelos com áudio (editável, ao lado do exe)
├── .github/workflows/     # ci.yml (testes) e release.yml (matriz de SO + publicação)
├── transcritor/
│   ├── config.py          # configuração em runtime (derivada das settings)
│   ├── settings.py        # configurações persistidas (JSON no perfil do usuário)
│   ├── ollama_manager.py  # localizar/iniciar/parar Ollama, baixar instalador
│   ├── ffmpeg_manager.py  # resolver (config → PATH → pasta local) e baixar o FFmpeg
│   ├── catalog.py         # lê audio_models.txt e mescla com os modelos instalados
│   ├── envutil.py         # ambiente limpo para subprocessos (remove vestígios do PyInstaller)
│   ├── webserver.py       # modo servidor: ServerConfig, login Google, rotas de job com dono, limpeza/TTL
│   ├── version.py         # __version__ (fonte única da versão/tag)
│   ├── setup.py           # preparo na inicialização (FFmpeg, Ollama, modelo base)
│   ├── media.py           # classificação áudio/vídeo + ffprobe
│   ├── audio.py           # extração, leitura PCM e divisão em trechos
│   ├── ollama_client.py   # cliente HTTP do Ollama
│   ├── srt.py             # montagem de cues e formatação SRT
│   ├── pipeline.py        # orquestra as etapas, reporta progresso
│   ├── jobs.py            # JobManager (estado, cancelamento)
│   └── browse.py          # navegador de pastas do servidor
├── templates/index.html
├── static/{css,js,img,vendor}
├── tests/
└── docs/
```

## 3. Pipeline

1. **Validação** — arquivo existe e é legível; `ffprobe` confirma fluxo de áudio; tipo = vídeo se houver fluxo de vídeo real (exclui capa de álbum `attached_pic`), senão áudio. (0–5 %)
2. **Extração** — `ffmpeg -vn -ac 1 -ar 16000 -c:a pcm_s16le` para um WAV temporário; serve para vídeo e áudio (normaliza qualquer formato). (5–15 %)
3. **Divisão em trechos (VAD)** — o PCM é lido com `numpy`; um VAD por energia (`audio.detect_speech`, sem dependências extras, limiar adaptativo) localiza as falas; `audio.split_by_speech` pula o silêncio e agrupa falas até ~12 s (máx. 28 s), cortando **apenas nas pausas**. Uma fala contínua maior que o máximo é dividida no ponto de menor energia. Com `Config.vad = False` volta ao corte por tempo (`split_chunks`). Como cada trecho começa e termina na fala, as legendas ficam mais bem temporizadas. (15 %)
4. **Transcrição (paralela)** — cada trecho vira um WAV em memória (base64) enviado ao `/api/chat` do Ollama no campo `images`, com `temperature=0` e `think=false`; o prompt pede só o texto falado, no idioma original. Até `parallel` trechos (padrão 2, máx. 4) vão ao servidor ao mesmo tempo; os resultados são reordenados pelo índice do trecho. Se as 3 primeiras respostas falharem, o trabalho aborta; falhas isoladas viram aviso. O ganho real depende do Ollama aceitar requisições simultâneas (`OLLAMA_NUM_PARALLEL`) e da VRAM livre; com `parallel = 1` o comportamento é sequencial. Todas as chamadas levam `keep_alive` (padrão `30m`, ajustável por `MT_KEEP_ALIVE`) para o modelo não ser descarregado entre trechos e trabalhos (o padrão do Ollama é 5 min). No início do trabalho o modelo é pré-aquecido (`/api/generate` sem prompt) em segundo plano, enquanto o ffmpeg extrai o áudio; falha no pré-aquecimento é ignorada. (15–75 %)
4b. **Tradução em lote** — as falas consecutivas são agrupadas (`translate_batch`, padrão 8 falas / ~2500 caracteres) e traduzidas numa única chamada **somente de texto**, que devolve um array JSON com o mesmo número de itens (as falas se dão contexto mútuo). Se a resposta vier fora do formato, cada fala do lote é traduzida individualmente; se isso também falhar, o texto original é mantido com aviso. Os lotes também rodam em paralelo. (75–97 %)
5. **Montagem do SRT** — o texto de cada trecho é dividido em frases/linhas curtas (≤ ~84 caracteres por cue) e os tempos são distribuídos proporcionalmente à duração do trecho; gravação em `<nome>.<idioma>.srt` ao lado da origem (fallback: pasta `output/`). (97–100 %)

### Motor "Qualidade" (protótipo, `transcritor/quality_engine.py`)

Terceiro caminho do pipeline (`job.engine = "quality"`), além do Ollama e dos motores alternativos. Desenho, papéis e diagramas em
[proposta-multiagente.md](proposta-multiagente.md); resultados em [modelos.md](modelos.md) seção 7. Fases (GPU única):

1. **Geração A** — modelo de áudio do Ollama (o escolhido na tela) transcreve todos os fragmentos (`_transcribe_all`).
2. **Geração B** — outro motor em todos os fragmentos: padrão OpenAI Whisper `large-v3-turbo` (`quality_b_engine`/`quality_b_model`); se faltar, um 2º modelo do Ollama (`gemma4:e2b`).
3. **Portão** (código) — marca os fragmentos em que o WER entre A e B passa de `quality_threshold` (padrão **0,05**: mais de ~5 % das palavras diferem, com números por extenso) ou em que só uma das gerações tem texto.
4. **Revisor** — modelo de áudio do Ollama (`quality_reviewer`; vazio = o do A) **reouve** só os fragmentos marcados e devolve JSON (`verdict` A/B/merge, `text`, `confidence`) por esquema (`OllamaClient.review`).
5. **Juiz** (código, com travas): confiança ≥ 0,5; a "mescla" só pode usar palavras de A, B ou do glossário e ficar a ≤ 30 % de uma das versões; qualquer falha mantém a geração prioritária (`quality_prefer`: A por padrão).

Depois seguem a limpeza, a tradução (pulada se o idioma já for o de saída) e as regras de legenda, como no caminho do Ollama. **Fragmentos:** salvos em
`work/<id>/` (`workdir.py`: WAVs + `manifesto.json` com A, B, revisão e origem do texto final) e apagados **só quando o trabalho conclui ou dá erro**; ao
cancelar ficam, e `workdir.purge_old` apaga pastas com mais de 7 dias (`MT_KEEP_WORK=1` mantém também ao concluir, para análise). Na tela, "Qualidade" é uma opção da
lista de motores (os modelos listados são os do Ollama, para a geração A); B, revisor e limiares são padrões da configuração (`Config.quality_*`). Há **modos de
qualidade com modelos fixos** (`quality_engine.MODES`; o modelo da tela é ignorado): o **2** = WhisperX + gerador `gemma4:e4b` + revisor `gemma4:12b` (v2.1, 2 passadas, especialistas de termos, pontuação e tradução); o **1** = autorrefinamento gerador↔revisor só com modelos do Ollama (`quality_self`): porteiro de fala (Silero, `speechgate.py`) descarta fragmentos sem fala antes do gerador `gemma4:e4b`, o revisor `gemma4:12b` ouve cada fragmento e o gerador refaz o que ele apontou (2 passadas; a votação entre hipóteses, `vote.py`, existe mas fica desligada). Medição em [experimento-modo1-v2.md](experimento-modo1-v2.md). Cada passada lê os prompts de `prompts/` (`transcritor/prompts.py`). Falta a camada de orquestração com o ADK.

### Regras de legenda, limpeza e contexto

- **Regras de legenda** (`srt.py`, Netflix *Portuguese (EMEA) Timed Text Style Guide*, conferido em 2026-11-10): linhas de até **42 caracteres**, no máximo **2 linhas** (uma só quando couber), quebra em pirâmide com a linha de baixo maior, sem deixar 1–2 palavras em cima nem terminar a linha de cima em artigo/preposição, preferindo cortar após pontuação; até **17 caracteres/s**. A página **não** define duração mínima/máxima, intervalo mínimo nem extensão do tempo final: usamos valores usuais da indústria (mín. 0,83 s, máx. 7 s, intervalo 2 quadros a 24 fps ≈ 0,083 s, extensão até 0,5 s), constantes no topo de `srt.py`. `apply_style` aplica tudo ao final; legendas ainda acima de 17 cps viram um aviso.
- **Limpeza do texto** (`quality.py`, sem modelo): descarta frases-fantasma que o modelo inventa em trechos sem fala ("Legendas pela comunidade Amara.org", "Obrigado por assistir"… só se forem o texto inteiro), texto longo demais para a duração (>45 caracteres/s), colapsa laços de repetição (4+ vezes seguidas) e remove repetição de 4+ palavras na emenda entre trechos. Tudo vira aviso no trabalho.
- **Contexto/glossário** (opcional, até 500 caracteres): tema, nomes e termos entram no prompt de transcrição e de tradução do Ollama e como `initial_prompt` nos motores Whisper.
- **Tempos**: cada trabalho cronometra modelo, extração, VAD, transcrição, tradução e montagem (`job.timings`) e mostra o RTF. Ver [avaliacao.md](avaliacao.md).

### Motores alternativos (`transcritor/engines.py`)

O pipeline escolhe o motor pelo `job.engine`. O Ollama (padrão) segue o caminho descrito acima. Os demais implementam a interface `Engine` (`available`, `is_installed`, `ensure_model`, `transcribe`) e devolvem **segmentos com timestamps absolutos** (`Seg`) e o idioma detectado:

| Motor | Pacote | Como transcreve | Modelos |
|-------|--------|-----------------|---------|
| `openai-whisper` | `openai-whisper` | por trecho de fala (os do VAD), idioma fixado no 1º trecho com fala | `.pt` em `~/.cache/whisper` (download com SHA-256 verificado) |
| `faster-whisper` | `faster-whisper` | idem, em CTranslate2 (`float16` na GPU, `int8` na CPU) | repositórios do Hugging Face (cache padrão) |
| `whisper-cpp` (menu: **Whisper Nativo**) | nenhum pacote Python: executável `whisper-cli` do whisper.cpp; **um motor, backend automático** (GPU NVIDIA/CUDA quando houver, senão CPU) | áudio inteiro numa chamada (`-oj`, progresso por `-pp`), timestamps do modelo; cancelar mata o processo | GGML `ggml-<nome>.bin` do Hugging Face, em `<pasta de dados>/whispercpp-models` (SHA-256 conferido) |
| `whisperx` | `whisperx` (+ `faster-whisper`) | áudio inteiro: VAD próprio, lotes, **alinhamento por palavra** (wav2vec2) e **diarização** opcional (pyannote) | os mesmos do faster-whisper; modelos de alinhamento/diarização baixam sozinhos na 1ª vez |
| `nemo-speech-cpp` | nenhum pacote Python: executável `nemo-speech` (NeMo-Speech.cpp, release oficial; CUDA/Vulkan/Metal/CPU conforme a máquina) | Transcribe-30B: 1 arquivo por trecho do VAD, uma execução em lote; 0,6 B: janelas de até 1 h; cancelar mata o subprocesso | GGUF em `<pasta de dados>/nemocpp-models` (SHA-256 e revisão fixos); ver [nemotron-nemo-speech.md](nemotron-nemo-speech.md) |
| `nemotron-vllm` | nenhum: cliente HTTP de um servidor vLLM (NVFP4, GPU Blackwell) | um pedido `/v1/chat/completions` por trecho do VAD | servidos pelo próprio vLLM |

Fluxo para um motor alternativo: `_ensure_engine_model` (baixa só se selecionado e ainda ausente; no servidor nunca baixa) → extração e VAD (igual) → `engine.transcribe` (15–75 %) → tradução **em lote pelo Ollama** (75–97 %), pulada se o idioma detectado já for o de saída; sem nenhum modelo no Ollama, mantém o idioma original com aviso → cues a partir dos segmentos (timestamps do motor) com prefixo `[Falante N]` quando o falante muda.

**whisper.cpp (um motor, dois executáveis):** o motor `whisper-cpp` escolhe o backend pela máquina (`WhisperCppEngine.choose_backend`: CUDA se houver GPU NVIDIA e binário CUDA, senão CPU; o dispositivo das Configurações pode forçar a CPU), cada backend com o seu executável em `<app>/whispercpp/cpu/` e `<app>/whispercpp/cuda/` (`transcritor/whispercpp_manager.py`). Ao gerar uma legenda com o motor (modo local), se o executável faltar, o app o **baixa do release oficial do GitHub** (CPU ~9 MB; CUDA ~685 MB, com as bibliotecas CUDA 12.4), conferindo o SHA-256 que o GitHub publica, antes de baixar o modelo GGML (compartilhado pelas duas variantes). Os nomes dos assets variaram entre releases, então há vários padrões por variante e vale o release mais novo que a tenha; para CUDA prefere-se a série 12.x. **Só há binário CUDA pronto para Windows x64** (Linux/macOS: compile); GPU Intel/AMD (Vulkan, SYCL) também exige compilar. A ordem de busca de cada variante é: Configurações → `WHISPER_CPP_PATH` / `WHISPER_CPP_CUDA_PATH` → pasta `whispercpp/<variante>/` → (só CPU) PATH. O backend CPU roda com `-ng` (sem GPU); o CUDA usa a GPU. O plano de hardware da tela e o log dizem qual backend foi usado (e por que caiu para a CPU, se for o caso). GPU Intel/AMD não é usada (sem binário pronto). No modo servidor nada é baixado por usuários. Não usa PyTorch nem Python, então cabe num executável leve.

Os pacotes são importados só ao usar o motor, então o app base e o executável continuam leves. **O executável PyInstaller não inclui os motores**: para usá-los, rode a partir do código-fonte com `pip install -r requirements-engines.txt` (instale só o que for usar). **Dispositivo** (`engines.pick_device`): `auto` escolhe CUDA > Intel XPU > Metal (MPS) > CPU; pode ser fixado em *Configurações* (campo *Dispositivo*), em `Config.device` ou na variável `MT_DEVICE` (a configuração explícita vence a variável). Quem pode usar o quê:

| Motor | CPU | CUDA (NVIDIA) | XPU (Intel) | Metal (Apple) | Precisão |
|-------|-----|---------------|-------------|---------------|----------|
| OpenAI Whisper (PyTorch) | sim | sim | sim* | sim* | `fp16` só no CUDA; nos demais, `float32` |
| faster-whisper (CTranslate2) | sim (`int8`) | sim (`float16`) | **não** | **não** | — |
| WhisperX | reconhecimento: CPU/CUDA; alinhamento e diarização (PyTorch): qualquer um | | | | |

\* exige a build do PyTorch com XPU (`+xpu`) ou MPS. Pedir um dispositivo que a máquina não tem é erro com a dica de instalação; pedir um que só o motor não suporta cai para a CPU com aviso; no automático, avisa se havia GPU que o motor não aproveita. No WhisperX, se o alinhamento ou a diarização falharem no XPU/Metal, são refeitos na CPU. `GET /api/devices` lista o que a máquina oferece (a tela mostra quando um motor Whisper é escolhido).
**Não validado em hardware real**: a lógica é testada com PyTorch/pacotes simulados (`tests/test_devices.py`); falta rodar `tools/validate_engines.py` em cada tipo de dispositivo.

API: `GET /api/engines` (lista, disponibilidade e dica de instalação), `GET /api/models?engine=<id>` (catálogo com `installed`), e `engine`/`diarize` no `POST /api/jobs`. No modo servidor só modelos já baixados são oferecidos.

O cancelamento é cooperativo: o pipeline verifica um `threading.Event` entre etapas e trechos.

> **Evolução possível:** hoje o pipeline é acoplado ao `OllamaClient`. Backends de ASR dedicado (whisper.cpp, faster-whisper/WhisperX, NeMo Parakeet) exigiriam uma interface de backend; a análise de viabilidade está em [pre-analise-asr.md](pre-analise-asr.md) e o plano em [otimizacoes-futuras.md](otimizacoes-futuras.md).

## 3.1 Decisões

| Decisão | Motivo |
|---------|--------|
| Capability `audio` via `/api/show` | É o sinal oficial do Ollama de que o modelo aceita áudio; evita lista fixa de nomes. |
| Áudio no campo `images` do `/api/chat` | Mecanismo aceito pelo Ollama para entradas de mídia (validado no ambiente alvo). |
| Trechos de ~12 s | Modelos de áudio têm limite curto por entrada; em testes, trechos curtos alucinaram menos e dão progresso e tempos mais finos. |
| Transcrever e depois traduzir | Testes mostraram que pedir tradução no próprio prompt de áudio é ignorado pelo modelo; a etapa textual garante o idioma de saída. |
| Corte em pausas | Reduz palavras partidas na fronteira dos trechos. |
| Threads + polling | Simples, sem dependências (sem Celery/WebSocket); adequado a uso local. |
| Caminho em vez de upload | Requisito do produto; evita copiar arquivos grandes. Navegador de pastas supre a ausência de caminho em `<input type=file>`. |
| Bootstrap local (`static/vendor`) | Funciona offline. |
| Ollama como requisito, instalador baixado no 1º uso | Mantém o executável leve e respeita a instalação existente do usuário. |
| Pasta padrão de modelos do Ollama | Quem tem modelos já tem Ollama; evita baixar de novo e duplicar GBs. |
| Pasta (`onedir`) em vez de arquivo único | Inicia mais rápido (sem extrair a cada execução) e é adequado a zip de *release*. |

## 4. API HTTP

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/` | Página principal |
| GET | `/api/models` | `{ok, models:[{name,size}], error?}` — só modelos com `audio` |
| GET | `/api/setup` | Estado do preparo: `{status, message, progress, ...}` |
| POST | `/api/setup/run` | Refaz a verificação/preparo (botão *Verificar novamente*) |
| POST | `/api/setup/install-ollama` | Baixa o instalador oficial do Ollama e o executa (Windows) |
| GET / POST | `/api/settings` | Lê / grava configurações (host, modelo base, download automático, trechos) |
| GET | `/api/browse?path=` | Lista pastas e arquivos de mídia de um diretório |
| GET | `/api/languages` | Idiomas de saída disponíveis |
| POST | `/api/jobs` | Corpo `{path, language, model}` → `{id}` (400 em validação) |
| GET | `/api/jobs/<id>` | `{status, progress, stage, message, warnings, output_path, error}` |
| POST | `/api/jobs/<id>/cancel` | Solicita cancelamento |
| GET | `/api/jobs/<id>/download` | Baixa o `.srt` |

`status`: `queued | running | done | error | cancelled`.

`GET /api/models` → `{ok, models:[{name, installed, recommended, size, approx_size, note}]}`: catálogo (ordem do arquivo) + instalados fora do catálogo, **instalados primeiro**. Em `POST /api/jobs`, o modelo pode ser instalado ou estar no catálogo; se não estiver instalado, o job começa baixando-o (primeiros 30 % da barra) e confirma a capability `audio` ao final.

## 4.1 Preparo na inicialização (`setup.py`)

Executado em thread ao subir o app (e a cada *Verificar novamente*). `status`: `checking | ffmpeg_missing | ollama_missing | installing | starting | pulling | ready | error`.

```
FFmpeg: config → PATH → pasta local ffmpeg/ ── achou ─► usa (todas as chamadas)
   │ caminho configurado inválido ─► ffmpeg_missing (sem fallback silencioso)
   ├ Windows ─► installing_ffmpeg: baixa gyan.dev "essentials" e extrai ffmpeg/ffprobe em ffmpeg/
   └ Linux/macOS ─► ffmpeg_missing (mostra o comando de instalação)
Ollama responde em <host>? ── sim ─┐
   │ não                           │
   ├ host remoto ─► error          │
   ├ binário local encontrado ─► inicia `ollama serve` (processo do app) ─► aguarda
   └ sem binário ─► ollama_missing (oferece instalador oficial)
Há modelo com capability audio? ── sim ─► ready
   │ não
   ├ download automático ligado ─► /api/pull do modelo base (progresso) ─► ready
   └ desligado ─► error (instrui a baixar um modelo)
```

- O binário é procurado no PATH e nos locais padrão (Windows: `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`; macOS: `/usr/local/bin`, `/Applications/Ollama.app`; Linux: `/usr/local/bin`, `/usr/bin`).
- O app **não define `OLLAMA_MODELS`**: usa a pasta padrão do Ollama, reaproveitando modelos existentes.
- O servidor só é encerrado ao sair se **foi o app que o iniciou**.
- O instalador (Windows) vem de `https://ollama.com/download/OllamaSetup.exe`, é baixado para a pasta temporária e executado; em outros sistemas a tela aponta para `https://ollama.com/download`.

## 5. Configuração

Persistida em `settings.json` (Windows `%LOCALAPPDATA%\MiniTranscritor`, macOS `~/Library/Application Support/MiniTranscritor`, Linux `~/.config/mini-transcritor`; sobrescrevível por `MINITRANSCRITOR_HOME`). Editável pela tela de configurações.

| Chave | Padrão | Uso |
|-------|--------|-----|
| `ollama_host` | `$OLLAMA_HOST` ou `http://127.0.0.1:11434` | URL do servidor Ollama |
| `ffmpeg_path` | *(vazio)* | pasta/exe do FFmpeg; vazio = automático (PATH → pasta local → download) |
| `base_model` | `gemma4:e4b` | modelo baixado se não houver nenhum com áudio |
| `auto_pull` | `true` | baixar o modelo base automaticamente |
| `chunk_seconds` | `12` (`$CHUNK_SECONDS`) | duração alvo dos trechos (5–28) |
| `parallel` | `2` | chamadas simultâneas ao Ollama (1–4) |
| `hf_token` | *(vazio; ou `$HF_TOKEN`)* | token do Hugging Face para a diarização do WhisperX; nunca é devolvido ao navegador |
| `translate_batch` | `8` | falas por chamada de tradução (1–16; 1 = fala a fala) |

Variáveis de ambiente: `OLLAMA_TIMEOUT` (padrão 600 s) e `HOST` / `PORT` (padrão `127.0.0.1` / `5000`; o launcher usa outra porta livre se a 5000 estiver ocupada).

## 5.1 Distribuição (PyInstaller, modo pasta)

`python build.py` gera `dist/MiniTranscritor/` com `MiniTranscritor.exe` e `_internal/` (Python, Flask, numpy, templates e estáticos). O `launcher.py` sobe o servidor Flask, abre o navegador, mantém uma janela de console com logs (fechá-la encerra o app) e, ao sair, para o Ollama que ele mesmo iniciou. Para o *release* no GitHub, compacte a pasta em zip. O FFmpeg não é empacotado (~230 MB no Windows): é detectado ou baixado para `ffmpeg/` ao lado do executável (`--with-ffmpeg` o inclui na build). O Ollama é requisito externo. `audio_models.txt` é copiado ao lado do executável (editável) e embutido como cópia de reserva.

**Subprocessos** (`ollama`, `ffmpeg`, `ffprobe`) rodam com `envutil.clean_env()`: o PyInstaller antepõe `_internal` ao `PATH`/`LD_LIBRARY_PATH`, e sem a limpeza o Ollama iniciado pelo app carregava DLLs do nosso pacote (e travava a pasta `dist`).

## 5.2 Catálogo de modelos (`audio_models.txt`)

Arquivo de texto, uma linha por modelo: `nome | tamanho aproximado | observação` (`#` = comentário). A ordem é a prioridade (Gemma primeiro). Procurado ao lado do executável / na raiz do projeto, depois na cópia embutida, depois num padrão mínimo no código. Atualização: editar o arquivo e reiniciar o app. **Regra de nomes:** só tags explícitas e descritivas (`gemma4:e4b`, `nemotron3:33b-q4_K_M`); linhas sem tag ou com `latest` são ignoradas por `catalog.parse`, e `Settings.base_model` recusa esses nomes (`catalog.has_explicit_tag`). Um teste automatizado garante que o catálogo distribuído cumpre a regra. Nomes instalados são comparados normalizando `:latest` apenas para casar modelos que o usuário já tinha.

## 5.2.1 Modo servidor (web)

Mesmo código, outro ponto de entrada: `create_app(server=ServerConfig)` (via `server.py`) registra `webserver.register(...)`; sem `server`, o app é o local. Documentação completa: [servidor.md](servidor.md). Resumo técnico:

- **Rotas:** no servidor *não existem* `/api/browse`, `/api/settings` e `/api/setup/install-ollama` (404). `POST /api/jobs` passa a ser multipart (upload); surgem `GET /api/jobs/current`, `DELETE /api/jobs/<id>` (limpar), `/login`, `/login/google`, `/auth/callback`, `POST /logout`, `/healthz`. `GET /api/models` devolve só instalados.
- **Guarda de acesso** (`before_request`): sem sessão → 401 (API) ou redirecionamento para `/login`; escritas exigem `X-Requested-With: fetch`; cada job tem `owner` (e-mail) e só o dono o acessa.
- **Fila:** `JobManager` com um worker e `queue.Queue` (FIFO); `describe()` informa `queue_position`; cancelar um job enfileirado o encerra na hora.
- **Arquivos:** `upload/<id>.<ext>` → apagado por `on_finish` ao fim de qualquer job; `download/<id>.srt` → apagado ao baixar (`remove_job`), ao `DELETE`, ou pelo *sweeper* (TTL). `upload/` e `download/` são recriados vazios na inicialização.
- **Estado em memória:** um único processo (waitress, 8 threads). Escalar horizontalmente exige a Fase 2 (fila/banco externos).
- **Segurança:** `authlib` valida o `id_token` do Google; `email_verified` obrigatório; lista de autorizados obrigatória (falha fechada); cookie `HttpOnly`/`SameSite=Lax`/`Secure`(https); `MAX_CONTENT_LENGTH` e limite de duração (checado com ffprobe).

## 5.3 Release e multiplataforma

PyInstaller não faz cross-compile; cada plataforma é compilada em runner nativo (matriz no GitHub Actions). Estratégia, comandos e o que foi/não foi testado: [release.md](release.md).

## 6. Tratamento de erros

Exceções de domínio (`MediaError`, `OllamaError`) viram `status=error` com mensagem amigável. Falha em um trecho gera aviso e segue; se **todos** falharem, o job termina em erro. Arquivos temporários são removidos em `finally`.

## 7. Testes

- Unitários: classificação de mídia, SRT, divisão de trechos, filtragem de modelos.
- Integração: Flask `test_client` com Ollama simulado e áudio/vídeo sintéticos gerados com ffmpeg.
- Ponta a ponta manual: Ollama real com modelo `audio`.


## Cancelamento e dispositivos (2026-11-12)

- **Cancelar é imediato.** `JobManager.cancel` marca o job como `cancelled` na hora (a tela se limpa e volta ao formulário), corta as chamadas em andamento ao Ollama (`OllamaClient.abort_all`: quem espera recebe `Cancelled` sem depender do sistema operacional acordar um socket bloqueado, e o socket é fechado para o Ollama parar de gerar), descarrega os modelos da GPU em segundo plano e deixa o worker sair em segundo plano. Um novo trabalho enviado nesse intervalo fica "Aguardando o cancelamento anterior terminar". Erros causados pelo corte contam como cancelamento, não como erro.
- **Chamadas longas que não dá para interromper por dentro** (PyTorch/CTranslate2: carregar modelo, reconhecer o arquivo inteiro no WhisperX, alinhar, diarizar) rodam em `engines._interruptible`: o worker sai na hora, mas o cálculo em si termina sozinho em segundo plano (o resultado é descartado; a CPU só fica livre quando ele acaba). Isolar o trabalho pesado em um subprocesso que possa ser encerrado fica como melhoria futura.
- **Tela dos motores Whisper:** a descrição do motor (o que ele faz) fica separada do bloco "Onde vai rodar nesta máquina", que mostra, por etapa (`Engine.stages`), o dispositivo em uso e por que não é a melhor GPU (`engines.engine_plan`, `GET /api/devices?engine=`), mais o hardware detectado e o dispositivo escolhido em Configurações.

## Log em arquivo, contexto do Ollama e teto de geração

- **Log** (`transcritor/applog.py`): `RotatingFileHandler` em `<pasta de dados>/logs/mini-transcritor.log` (5 MB x 5), ligado em `create_app`. `pipeline.run_job` registra início (arquivo, motor, modelo, idioma), fim (mensagem, tempos por etapa, saída), cancelamento, erro (com traceback quando inesperado) e **todos os `job.warnings`**. A tela de conclusão não lista mais avisos (eles seguem em `job.warnings` na API e no console do navegador). Detalhes por fragmento, como o motivo de cada revisão que falhou, vão ao log em tempo real; em `job.warnings` fica **uma** linha com a contagem. O nome do arquivo de mídia é registrado; as falas, não.
- **Mensagem de conclusão:** "Tempo de processamento: hh:mm:ss" + caminho da legenda; RTF, nº de legendas e segundos de áudio ficam só em `job.timings` / `job.audio_seconds`.
- **Investigação das "revisões que falharam"** (2026-11-12): (1) `OllamaClient.transcribe` não tinha `num_predict`; em música/ruído o modelo gerava ~30 mil tokens em laço, e esse texto, entregue ao revisor, estourava o contexto ("request (33107 tokens) exceeds the available context size (32768)"), o que explica o 400. Agora o teto é proporcional à duração do trecho (`token_cap`: 12 tokens/s + 96, entre 128 e 1024) e as traduções/pontuação têm `text_cap`. (2) "A revisão não devolveu JSON válido" era a resposta do revisor **cortada em 512 tokens** (o campo `analysis` passava do "máximo de 80 palavras" do prompt). Os esquemas agora limitam cada campo (`maxLength`/`maxItems`; o Ollama gera com gramática, então o JSON sempre fecha) e `MAX_JSON_TOKENS` subiu para 768.
- **Janela de contexto:** o app **não fixava** o contexto (só a triagem pedia 12 288, o que ainda forçaria recarregar o modelo). O valor efetivo era o padrão do Ollama para a GPU (32 768 nesta máquina, `gemma4:12b` em `/api/ps`). Agora **toda** chamada leva `num_ctx = 32768` (`OllamaClient.num_ctx`; variável `MT_NUM_CTX` para mudar): o mesmo valor em tudo, sem recarga entre chamadas e sem depender da VRAM da máquina.
