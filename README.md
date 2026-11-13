# Mini Transcritor

Aplicação web local (Python/Flask) que gera **legendas SRT** a partir de um arquivo de **áudio ou vídeo**, usando apenas **modelos multimodais com suporte a áudio do Ollama**. Tudo roda na sua máquina.

- Entende vídeo (mkv, avi, mp4, mov, webm…) e áudio (mp3, flac, wav, m4a, ogg, opus…).
- Vídeo → extrai o áudio → mesmo pipeline do áudio.
- Lista apenas modelos locais do Ollama com capability `audio`.
- Escolha do idioma da legenda e barra de progresso real.
- Interface Bootstrap simples (funciona offline).

## Requisitos

| Item | Observação |
|------|-----------|
| [Ollama](https://ollama.com/download) | **Requisito.** Se não estiver instalado, o app avisa e oferece baixar o instalador oficial (Windows). Se estiver instalado mas parado, o app o inicia. |
| [FFmpeg](https://ffmpeg.org) | Usa o do sistema (ou o caminho definido em Configurações); se não houver, **baixa para a pasta `ffmpeg/` do app** (Windows). Linux/macOS: `apt install ffmpeg` / `brew install ffmpeg`. |
| Modelo com áudio | Se nenhum modelo local tiver a capability `audio`, o app baixa o modelo base **`gemma4:e4b`** (≈ 6,6 GB) na inicialização. Modelos já existentes na pasta padrão do Ollama são reaproveitados. |
| Python ≥ 3.10 | Apenas para rodar a partir do código-fonte |

## Dois modos: local (desktop) e servidor (web)

| Quero… | Comando | Modo |
|--------|---------|------|
| Usar no meu computador (arquivo por caminho, sem login) | `python app.py` ou `MiniTranscritor.exe` | local |
| Publicar na web (login Google, upload, fila, arquivos temporários) | `python server.py` ou `docker compose up -d` | servidor |
| Gerar o executável | `python build.py` / `python release.py` | só local |

O mesmo código atende os dois; o modo é só o ponto de entrada. Guia do servidor: [docs/servidor.md](docs/servidor.md). As credenciais do Google ficam no arquivo `.env` (não versionado; modelo em `.env.example`), que o `server.py` lê sozinho.

## Usando o executável (Windows)

Baixe o arquivo da sua plataforma em *Releases* (`windows-x64`, `linux-x64`, `linux-arm64`, `macos-arm64`), extraia e execute `MiniTranscritor\MiniTranscritor.exe` (Linux/macOS: `./MiniTranscritor`). Uma janela de console (logs) abre junto com o navegador; **fechá-la encerra o app** (e o Ollama, se foi o app que o iniciou).

## Executando a partir do código

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Abra **http://127.0.0.1:5000**.

## Modelos

A lista de modelos vem de [audio_models.txt](audio_models.txt) (catálogo estático, família Gemma primeiro; edite para atualizar). Os nomes usam sempre a **tag descritiva** (parâmetros e quantização, ex.: `nemotron3:33b-q4_K_M`), nunca `latest`. Qualidade medida: [docs/modelos.md](docs/modelos.md). Os **instalados** aparecem destacados no topo; escolher um não instalado o baixa antes de gerar.

## Motores alternativos (opcionais)

Além do Ollama, é possível usar **OpenAI Whisper**, **faster-whisper** e **WhisperX** (timestamps reais; WhisperX com alinhamento por palavra e **diarização**). Instale só o que for usar: `pip install -r requirements-engines.txt` (veja o arquivo). O catálogo de modelos é [whisper_models.txt](whisper_models.txt); o modelo é baixado apenas quando você o seleciona. O **whisper.cpp** (no menu, *Whisper Nativo*) não usa Python nem PyTorch e é um motor só, com backend automático: GPU NVIDIA (CUDA) quando houver, senão CPU. O app baixa o executável do release oficial ao usar o motor (pastas `whispercpp/cpu/` e `whispercpp/cuda/`; binário CUDA só para Windows x64; GPU Intel/AMD não é usada). O executável do app não inclui os pacotes Python (use o código-fonte). Detalhes em [docs/arquitetura.md](docs/arquitetura.md).

Também há os motores da NVIDIA: **NeMo-Speech.cpp** (Nemotron 3.5 Transcribe 30B-A4B em duas quantizações e ASR Streaming 0,6 B; roda em CPU, CUDA ou Vulkan, inclusive Intel Arc) e o cliente de um servidor **vLLM** para o NVFP4 (GPU Blackwell). Detalhes e medições: [docs/nemotron-nemo-speech.md](docs/nemotron-nemo-speech.md).

**Log:** avisos e detalhes técnicos (revisões que falharam, repetições removidas, leitura rápida...) não aparecem mais na tela de conclusão; ficam no arquivo `mini-transcritor.log` em `<pasta de dados>/logs/` (Windows: `%LOCALAPPDATA%\MiniTranscritor\logs\`), rotativo (5 MB x 5), com início, fim, tempos e erros de cada trabalho.

## Gerando o executável e o release

```bash
python build.py                 # dist/MiniTranscritor/ (somente o SO em que você roda)
python release.py               # dry-run: testes -> build -> smoke test -> zip/tar.gz + sha256
python release.py --publish     # idem + gh release create (pede confirmação)
```

O release **oficial**, com todas as plataformas, sai do GitHub Actions ao enviar uma tag `vX.Y.Z` — veja [docs/release.md](docs/release.md). O trabalho segue o fluxo de duas branches (`dev` e `main`) descrito em [docs/fluxo-git.md](docs/fluxo-git.md).

## Configuração

Pelo botão **Configurações** da interface (servidor Ollama, modelo base, download automático, duração dos trechos). Salvo em `%LOCALAPPDATA%\MiniTranscritor\settings.json` (macOS: `~/Library/Application Support/MiniTranscritor`; Linux: `~/.config/mini-transcritor`).

Variáveis de ambiente: `OLLAMA_HOST` e `CHUNK_SECONDS` (valores iniciais), `OLLAMA_TIMEOUT` (padrão 600 s), `HOST`/`PORT` (padrão `127.0.0.1`/`5000`; se a porta estiver ocupada o executável usa outra livre), `MINITRANSCRITOR_HOME` (pasta das configurações).

> Não exponha o app à rede (`HOST=0.0.0.0`): o navegador de pastas lê o disco do servidor.

## Testes

```bash
python -m pytest -q
```

Os testes usam um Ollama simulado e geram mídia sintética com o ffmpeg.

## Documentação

- [PRD](docs/prd.md) — requisitos e escopo
- [Arquitetura](docs/arquitetura.md) — componentes, pipeline e API
- [Mini-tutorial](docs/tutorial.md) — passo a passo de uso
- [Modo servidor (web)](docs/servidor.md) — login Google, upload, fila, Docker, HTTPS
- [Otimizações futuras](docs/otimizacoes-futuras.md) — roadmap: Whisper com timestamps (WhisperX, faster-whisper, whisper.cpp), interface de backend, Fase 2 do servidor
- [Pré-análise ASR](docs/pre-analise-asr.md) — NVIDIA NeMo (Nemotron/Parakeet/Canary) e família Whisper como possíveis backends futuros
- [Modelos](docs/modelos.md) — quais modelos entram na lista, por quê e como se saíram nos testes
- [Experimento com um longa-metragem (legenda pt-BR)](docs/experimento-longa-metragem.md) — pipeline completo em 30 min de filme: e4b, Whisper turbo, WhisperX com diarização e Qualidade
- [Relatório consolidado da avaliação](docs/relatorio-avaliacao.md) — resultados, correções, recomendação final
- [Experimento do modo de qualidade 1 (porteiro de fala, votação)](docs/experimento-modo1-v2.md)
- [Experimento do laço de refinamento e dos modos de qualidade](docs/experimento-loop-refinamento.md) — veredito, diagnóstico e como reproduzir; plano da v2 em [docs/plano-pipeline-v2.md](docs/plano-pipeline-v2.md); prompts dos agentes em [prompts/](prompts/README.md)
- [Proposta multiagente (opção Qualidade)](docs/proposta-multiagente.md) — desenho textual e visual, agentes, travas, evidências e plano
- [Validação e medição](docs/avaliacao.md) — roteiro para validar os motores e medir velocidade/qualidade com a GPU livre
- [Release](docs/release.md) — estratégia de versões, builds multiplataforma e publicação

## Limitações

- Com o motor Ollama, os tempos do SRT são aproximados (baseados nos trechos de áudio, não em palavras), pois modelos multimodais não fornecem timestamps confiáveis.
- A fidelidade depende do modelo; revise a legenda. Em testes, modelos maiores não foram necessariamente melhores.
- O idioma de saída é obtido transcrevendo e, em seguida, traduzindo o texto com o mesmo modelo.
