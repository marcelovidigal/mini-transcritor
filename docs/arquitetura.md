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
3. **Divisão em trechos** — o PCM é lido com `numpy`; alvo de ~12 s por trecho (máx. 28 s); o ponto de corte é o quadro de menor energia na janela final do trecho, evitando cortar palavras. Trechos silenciosos são descartados. (15 %)
4. **Transcrição** — cada trecho vira um WAV em memória (base64) enviado ao `/api/chat` do Ollama no campo `images` da mensagem, com `temperature=0` e `think=false`. O prompt pede só o texto falado, no idioma original. Em seguida, uma chamada **somente de texto** ao mesmo modelo traduz o resultado para o idioma escolhido (mantém se já estiver nele); se a tradução falhar, o texto original é mantido com aviso. (15–97 %)
5. **Montagem do SRT** — o texto de cada trecho é dividido em frases/linhas curtas (≤ ~84 caracteres por cue) e os tempos são distribuídos proporcionalmente à duração do trecho; gravação em `<nome>.<idioma>.srt` ao lado da origem (fallback: pasta `output/`). (97–100 %)

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
