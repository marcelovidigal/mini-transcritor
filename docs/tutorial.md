# Mini-tutorial — Mini Transcritor

Em 5 minutos: do arquivo de vídeo/áudio ao `.srt`.

## 1. Requisitos

- **Ollama** instalado (<https://ollama.com/download>). Se faltar, o app mostra um aviso explicando que ele é um requisito e, no Windows, oferece **Baixar e instalar o Ollama**; depois de instalar, clique em **Verificar novamente**.
- **FFmpeg**: se já existir no sistema, é usado; se não, o app **baixa para a pasta `ffmpeg/`** ao lado do executável (Windows, ~115 MB, na primeira execução). Em Linux/macOS o app mostra o comando de instalação (`apt install ffmpeg` / `brew install ffmpeg`). Para usar outro FFmpeg, informe a pasta em *Configurações*.
- Um modelo com áudio. Você não precisa baixar nada: se o Ollama não tiver nenhum modelo com capability `audio`, o app baixa o modelo base `gemma4:e4b` (≈ 6,6 GB) sozinho, com barra de progresso. Se você já tem modelos (por exemplo `gemma4:e2b`), eles são reaproveitados.

## 2. Iniciar o app

- **Executável:** abra `MiniTranscritor\MiniTranscritor.exe`. Para encerrar, feche a janela de console.
- **Código-fonte:** `pip install -r requirements.txt` e `python app.py`.

O navegador abre em <http://127.0.0.1:5000> (ou em outra porta livre, informada no console). Na tela aparece um cartão **"Preparando o ambiente"** enquanto o app verifica FFmpeg e Ollama, inicia o servidor Ollama se ele estiver parado e, se preciso, baixa o modelo base. O formulário é liberado quando o preparo termina.

## 3. Gerar a legenda

1. **Arquivo** — digite o caminho completo (ex.: `D:\Videos\aula.mkv`) ou clique em **Procurar…** e navegue pelas pastas; só pastas e arquivos de mídia são exibidos. Clique no arquivo para selecioná-lo.
2. **Idioma da legenda** — escolha o idioma de saída (ex.: Português (Brasil)).
3. **Motor de transcrição** — *Ollama* (padrão) ou, se instalados, *OpenAI Whisper*, *faster-whisper* e *WhisperX* (ver `requirements-engines.txt`). Motor não instalado aparece com o comando `pip` para instalar. No WhisperX há ainda a caixa **Identificar quem fala** (exige o token do Hugging Face em Configurações). Nos motores Whisper a lista mostra **✔ Baixados nesta máquina** e **Disponíveis — serão baixados ao gerar**; o modelo só é baixado quando você o escolhe e clica em *Gerar legenda*.
3b. **Modelo** (Ollama) — lista só modelos com suporte a áudio, em dois grupos: **✔ Instalados nesta máquina** (no topo) e **Recomendados — serão baixados ao gerar**. A família Gemma vem primeiro. Se escolher um não instalado, ele é baixado antes de gerar (a barra mostra o download). Use ↻ após instalar modelos por fora.
3c. **Contexto ou glossário** (opcional) — tema, nomes próprios e termos técnicos (até 500 caracteres); ajuda o modelo a acertar o vocabulário.
4. Clique em **Gerar legenda**. A barra mostra a etapa atual: *validando → extraindo áudio → transcrevendo trecho N de M → gravando*.
5. Ao terminar, a mensagem mostra quantas legendas saíram e o tempo gasto (ex.: *42 s para 600 s de áudio*). Aparece o caminho do `.srt` (salvo **ao lado do arquivo original**, como `aula.pt-BR.srt`) e o botão **Baixar .srt**.

Botões extras: **Cancelar** (durante o processamento) e **Limpar** (zera o formulário).

## 4. Usar a legenda

Coloque o `.srt` na mesma pasta do vídeo com o mesmo nome base (ex.: `aula.mkv` + `aula.pt-BR.srt`); players como VLC e MPC carregam automaticamente, ou use *Legendas → Adicionar arquivo*.

## 5. Configurações

O botão **Configurações** (canto superior direito) permite ajustar:

| Campo | Para quê |
|-------|----------|
| FFmpeg (pasta ou executável) | Em branco = automático (sistema → pasta local → download). Informe a pasta que contém `ffmpeg` e `ffprobe` para forçar um FFmpeg específico; caminho inválido é recusado ao salvar. |
| Servidor Ollama | Endereço do Ollama (padrão `http://127.0.0.1:11434`). Se for um endereço local e o servidor estiver parado, o app o inicia; endereços remotos precisam estar ativos. |
| Modelo base | Modelo baixado na inicialização **somente** se não houver nenhum com áudio. |
| Baixar automaticamente | Desligue para nunca baixar nada sem você pedir. |
| Duração dos trechos | 5–28 s (padrão 12). |
| Dispositivo dos motores Whisper | *Automático* (padrão) escolhe CUDA > Intel XPU > Metal > CPU; ou fixe um. O faster-whisper só usa CPU ou CUDA. Dispositivo inexistente na máquina é recusado com o motivo. |
| Token do Hugging Face | Só para a diarização do WhisperX. Fica salvo localmente e nunca volta à tela; em branco = manter. |

Ao salvar, o app refaz a verificação com os novos valores.

## 6. Dicas e solução de problemas

| Sintoma | O que fazer |
|---------|-------------|
| "Ollama não encontrado" | Instale o Ollama (botão no próprio aviso ou <https://ollama.com/download>) e clique em *Verificar novamente*. |
| "Não foi possível conectar ao Ollama" | Confira o endereço em *Configurações*. |
| Lista de modelos vazia | Reinicie o preparo (*Configurações → Salvar*) ou baixe um modelo com capability `audio` e clique em ↻. |
| "FFmpeg não encontrado" / download falhou | Verifique a internet e clique em *Verificar novamente*, ou instale o FFmpeg e informe a pasta em *Configurações*. |
| "O arquivo não contém nenhuma faixa de áudio" | O vídeo não tem áudio. |
| Texto com erros | Tente outro modelo (maior nem sempre é melhor) ou reduza a duração dos trechos em *Configurações* (ex.: 8 s). Revise a legenda. |
| Demora | Depende do modelo e do hardware; cada ~12 s de mídia gera uma chamada ao modelo. |
| Tempos levemente fora | Esperado: os tempos seguem os trechos de áudio, não as palavras. |

Avisos e detalhes técnicos de cada trabalho ficam no arquivo de log, não na tela: `%LOCALAPPDATA%\MiniTranscritor\logs\mini-transcritor.log` (Windows). Abra-o ao investigar uma falha ou comparar execuções.

## 7. Atualizar a lista de modelos recomendados

Edite `audio_models.txt` (ao lado do executável) — uma linha `nome:tag | tamanho | observação` por modelo, em ordem de prioridade — e reinicie o app. Use sempre a **tag descritiva** (com a quantidade de parâmetros e a quantização, ex.: `nemotron3:33b-q4_K_M`), nunca `latest`; veja as tags em `https://ollama.com/library/<modelo>/tags`. Qualidade medida de cada modelo: [modelos.md](modelos.md). Veja novidades em <https://ollama.com/search?c=audio>.

## 8. Linux e macOS

Baixe o arquivo da sua plataforma na página de Releases, extraia e execute `./MiniTranscritor`. Instale antes o Ollama (`curl -fsSL https://ollama.com/install.sh | sh` no Linux; app em ollama.com/download no macOS) e o FFmpeg. No macOS, binários não assinados são bloqueados na primeira vez: `xattr -dr com.apple.quarantine MiniTranscritor/`.

## 9. Versão web (servidor)

Para publicar o app para outras pessoas (login com Google, upload, fila, arquivos temporários), veja [servidor.md](servidor.md). Para o usuário final, o uso é: entrar com Google → escolher o arquivo → **Gerar legenda** → acompanhar a fila/progresso → **Baixar .srt** (a tela volta ao início e o servidor apaga tudo).
