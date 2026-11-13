# PRD — Mini Transcritor

**Versão:** 1.0 · **Status:** Em implementação

## 1. Resumo

O **Mini Transcritor** é uma aplicação web local (Python/Flask) que gera legendas no formato **SRT** a partir de um arquivo de áudio ou vídeo existente no computador do usuário. A transcrição é feita exclusivamente por **modelos multimodais com capacidade de áudio, servidos por um servidor Ollama local**. Nenhuma outra ferramenta de transcrição (Whisper, APIs na nuvem etc.) é usada nesta versão; o único utilitário auxiliar é o `ffmpeg`/`ffprobe`, necessário para ler e extrair o áudio dos arquivos.

## 2. Problema e objetivo

Gerar legendas manualmente é lento, e serviços online exigem enviar a mídia a terceiros. O objetivo é oferecer um fluxo simples, 100% local e privado: escolher um arquivo, escolher idioma e modelo, clicar em *Gerar* e receber o `.srt`.

## 3. Público-alvo

Pessoas que já executam modelos no Ollama e querem legendar vídeos, aulas, entrevistas ou podcasts sem sair da máquina.

## 4. Escopo

### 4.1 Dentro do escopo (v1)

| ID | Requisito |
|----|-----------|
| RF-01 | Selecionar um arquivo local informando o **caminho** (digitado ou escolhido em um navegador de pastas do servidor). |
| RF-02 | **Identificar o tipo** do arquivo (áudio ou vídeo) em formatos padrão: vídeo — mkv, avi, mp4, mov, webm, flv, wmv, m4v, ts, mpg/mpeg; áudio — mp3, flac, wav, m4a, aac, ogg, opus, wma, aiff. A decisão final é confirmada com `ffprobe` (precisa existir fluxo de áudio). |
| RF-03 | Se for **vídeo**, extrair o áudio e seguir o mesmo pipeline do áudio. |
| RF-04 | Listar **somente modelos do Ollama que declaram a capability `audio`**, entre os modelos já baixados no servidor. |
| RF-05 | Escolher o **idioma de saída** da legenda. |
| RF-06 | Botão **Gerar legenda** com **barra de progresso** que reflete as etapas reais (validação, extração, transcrição por trechos, gravação). |
| RF-07 | Produzir um arquivo `.srt` válido (numeração, tempos `HH:MM:SS,mmm`, UTF-8) salvo ao lado do arquivo de origem e disponível para **download**. |
| RF-08 | Botão para **cancelar** o processamento em andamento. |
| RF-09 | Botão para **atualizar** a lista de modelos e **limpar** o formulário. |
| RF-10 | Mensagens de erro claras (Ollama offline, nenhum modelo com áudio, arquivo inválido, ffmpeg ausente). |
| RF-11 | **Ollama é requisito.** Na inicialização o app verifica se há um servidor Ollama acessível; se houver instalação local com o servidor parado, inicia-o. Se não existir Ollama, exibe uma tela informando que ele é obrigatório e oferece **baixar o instalador oficial** (Windows) ou abrir a página de download; o usuário instala e clica em *Verificar novamente*. |
| RF-12 | **Reaproveitar o Ollama existente**: usa a pasta padrão de modelos do Ollama (o app nunca altera `OLLAMA_MODELS`); modelos já baixados são reutilizados sem novo download. |
| RF-13 | **Modelo base**: na inicialização, se não houver nenhum modelo com capability `audio`, baixa automaticamente o modelo base (`gemma4:e4b`, ≈ 6,6 GB) mostrando barra de progresso do download. Se já houver ao menos um modelo com áudio, nada é baixado. |
| RF-14 | **Tela de configurações** (ícone de engrenagem): endereço do servidor Ollama, modelo base, download automático do modelo base (sim/não) e duração dos trechos. Salvas em arquivo JSON no perfil do usuário. |
| RF-19 | **Modo servidor (web)**: versão para publicar em servidor, ativada pelo ponto de entrada (`python server.py` / Docker); o modo local (`app.py`/executável) permanece inalterado. Detalhes em [servidor.md](servidor.md). |
| RF-20 | **Acesso por conta Google**: só e-mails/domínios autorizados entram (login OAuth); sem lista configurada o servidor não inicia; cada usuário vê apenas os próprios jobs. |
| RF-21 | **Upload e arquivos temporários**: o arquivo é enviado pelo navegador para a pasta `upload/` e **apagado ao fim do processamento**; a legenda fica em `download/` e é **apagada após o download** (ou ao limpar/trocar de arquivo, ou por TTL). |
| RF-22 | **Fila única**: um job processado por vez, em ordem de chegada, com posição na fila visível; uma legenda por usuário por vez; só modelos já instalados; limites de tamanho e de duração. |
| RF-23 | **Fluxo de tela no servidor**: ao concluir, o botão de download fica ativo enquanto o usuário não baixar, não limpar nem escolher outro arquivo; ao baixar, o download inicia e a tela volta ao estado inicial; recarregar a página retoma o processo. |
| RF-24 | **Motores de transcrição alternativos** (opcionais): além do Ollama (padrão), o usuário escolhe **OpenAI Whisper**, **faster-whisper** ou **WhisperX**. Cada motor tem catálogo estático (`whisper_models.txt`) com todas as versões de modelo (nomes específicos, sem apelidos); o modelo é **baixado só quando selecionado** para gerar, e a tela marca "baixado" apenas para o que já está no disco. Os pacotes são opcionais (aviso com o comando `pip` quando ausentes). Produzem timestamps reais por trecho de fala (por palavra no WhisperX); a tradução para o idioma de saída continua sendo feita por um modelo do Ollama, e é dispensada se o áudio já estiver nesse idioma. |
| RF-25 | **Diarização** (só WhisperX): opção "identificar quem fala" na tela de início; a legenda marca a troca de falante (`[Falante 1]`, `[Falante 2]`…). Exige token do Hugging Face (Configurações ou `HF_TOKEN`) e aceitar a licença do modelo pyannote. O token nunca volta ao navegador. |
| RF-26 | **Velocidade**: VAD (pula silêncio e corta só nas pausas), transcrição e tradução em paralelo (`parallel`), tradução em lote (`translate_batch`) e modelo mantido na memória (`keep_alive` + pré-aquecimento). Ver [arquitetura.md](arquitetura.md). |
| RF-27 | **Regras de legenda** (Netflix, português): até 42 caracteres por linha, 2 linhas, 17 caracteres/s, quebra em pirâmide sem separar artigo/preposição, duração e intervalos mínimos; avisa quando a leitura fica rápida demais. |
| RF-28 | **Qualidade do texto**: descarta frases inventadas e laços de repetição, remove repetição nas emendas e aceita contexto/glossário opcional (até 500 caracteres). |
| RF-29 | **Medição**: cada trabalho informa o tempo por etapa e o RTF; `tools/evaluate.py` mede WER, regras e erro de tempo ([avaliacao.md](avaliacao.md)). |
| RF-15 | **Distribuição**: pasta com executável (`MiniTranscritor.exe`) gerada com PyInstaller, que sobe o servidor, abre o navegador e encerra o que iniciou ao fechar; pronta para virar *release* no GitHub (zip da pasta). |
| RF-16 | **FFmpeg**: o app usa, nesta ordem, (1) o caminho informado em *Configurações*, (2) o FFmpeg/FFprobe do sistema (PATH), (3) a pasta `ffmpeg/` ao lado do app. Se nenhum existir, **baixa e instala localmente** em `ffmpeg/` (automático no Windows; em Linux/macOS exibe como instalar). Todas as chamadas usam o executável resolvido. Um caminho configurado inválido é reportado, sem cair silenciosamente em outro. |
| RF-17 | **Catálogo de modelos**: a lista de modelos vem de um arquivo de texto estático (`audio_models.txt`), com os modelos Ollama multimodais mais competentes para áudio (família **Gemma** primeiro), atualizável sem recompilar. Os **instalados na máquina são destacados** (grupo próprio, no topo); modelos de áudio instalados fora do catálogo também aparecem. Escolher um modelo não instalado o **baixa antes de gerar**, com progresso. **Nomes sempre com tag descritiva** (parâmetros e quantização, ex.: `nemotron3:33b-q4_K_M`), nunca `latest`: o catálogo ignora entradas sem tag e as configurações recusam `latest`. Resultados de testes de qualidade em [modelos.md](modelos.md). |
| RF-18 | **Multiplataforma e release**: executáveis para Windows x64, Linux x64, Linux arm64 e macOS (Apple Silicon), gerados em CI nativo; um release do GitHub por versão com um arquivo por plataforma; o release só ocorre por decisão explícita (tag `vX.Y.Z` ou execução manual com *publish*). Ver [release.md](release.md). |

### 4.2 Fora do escopo (v1)

- Outros backends além do Ollama (NeMo/Nemotron ASR, Whisper e derivados). Diretriz já decidida para o futuro: **dar preferência às variantes Whisper com timestamps** (WhisperX, faster-whisper, whisper.cpp). Ver [otimizações futuras](otimizacoes-futuras.md) e a [pré-análise](pre-analise-asr.md); gerenciamento avançado de modelos (listar/excluir/escolher outros para baixar) pela interface.
- Empacotar o Ollama ou o FFmpeg no executável: o Ollama é requisito instalado separadamente; o FFmpeg é detectado ou baixado para a pasta local no primeiro uso.
- Instalação automática do Ollama e do FFmpeg em Linux/macOS (o app mostra o comando de instalação).
- Assinatura de código / notarização (Windows SmartScreen e macOS Gatekeeper exibirão aviso em binários não assinados).
- Upload do arquivo pelo navegador (a seleção é por caminho local).
- Diarização de falantes, edição de legenda, legendas incrustadas (hardcoded).
- Contas/quotas próprias, fila persistente, armazenamento de objetos e planos de uso (Fase 2, público aberto). A Fase 1 (servidor privado com lista de autorizados) está no escopo (RF-19 a RF-23).

## 5. Fluxo do usuário

1. Executa `MiniTranscritor.exe` (ou `python app.py`); o navegador abre em `http://127.0.0.1:5000`.
2. **Primeira execução / preparo**: o app verifica FFmpeg e Ollama, inicia o servidor se necessário e, se não houver modelo com áudio, baixa o modelo base com barra de progresso. Se o Ollama não existir, orienta a instalação.
3. Informa/seleciona o arquivo, escolhe idioma e modelo.
4. Clica em **Gerar legenda** e acompanha a barra de progresso.
5. Ao concluir, baixa o `.srt` ou o localiza ao lado da mídia original.

## 6. Interface

Página única, Bootstrap 5 padrão, sucinta: logo + nome do app; campo de caminho + botão *Procurar*; seletor de idioma; seletor de modelo (+ atualizar); botões *Gerar legenda*, *Cancelar*, *Limpar*; barra de progresso com etapa atual; área de resultado/erro com link de download.

## 7. Requisitos não funcionais

- **Privacidade:** nada sai da máquina (exceto a comunicação com o Ollama configurado, por padrão `localhost`).
- **Robustez:** mídias longas são processadas em trechos curtos (os modelos de áudio têm limite de duração por entrada); falha em um trecho não derruba o job inteiro (o trecho é registrado e ignorado, com aviso final).
- **Segurança:** o app é pensado para uso local; escuta apenas em `127.0.0.1` por padrão. O navegador de pastas lê o sistema de arquivos do servidor, portanto não deve ser exposto à rede.
- **Portabilidade:** Windows, Linux e macOS com Python ≥ 3.10. Executáveis: Windows x64, Linux x64/arm64, macOS arm64 (Intel opcional, conforme runner disponível).
- **Testabilidade:** o cliente Ollama é isolado para permitir testes com dublê.

## 8. Premissas e limitações conhecidas

- O Ollama deve estar instalado; o app o inicia se necessário. Deve haver ao menos um modelo com capability `audio` (o app baixa o modelo base se não houver). Neste ambiente: família `gemma4` variantes `e2b`, `e4b`, `12b`. As variantes `26b`/`31b` do gemma4 **não** aceitam áudio.
- Os modelos não-Gemma do catálogo (`nemotron3`, `lfm3-*`) vêm da busca por capability *audio* na biblioteca do Ollama e **não foram testados** neste projeto.
- Modelos multimodais não retornam timestamps confiáveis. Os tempos do SRT derivam dos **limites dos trechos de áudio** (cortados preferencialmente em pausas), distribuindo o texto de cada trecho proporcionalmente ao seu tamanho. A precisão é de alguns segundos, não de palavra.
- O idioma de saída é obtido transcrevendo no idioma original e traduzindo o texto com o mesmo modelo; a qualidade da tradução depende do modelo.
- Modelos de áudio locais podem errar palavras ou alucinar, principalmente com fala ruidosa/sintética; modelos maiores nem sempre são melhores (no ambiente de teste `gemma4:e4b` superou o `12b`). Revise a legenda gerada.
- A qualidade e o tempo dependem do modelo e do hardware.

## 9. Critérios de aceite

1. Arquivo de áudio válido → `.srt` válido gerado e baixável.
2. Arquivo de vídeo (mp4/mkv/avi) → áudio extraído e `.srt` gerado.
3. A lista de modelos exibe apenas modelos com `audio`.
4. A barra de progresso avança monotonicamente de 0 a 100 %.
5. Erros previstos exibem mensagem compreensível sem quebrar a página.
6. Testes automatizados passam e um teste ponta a ponta real com Ollama foi executado.
