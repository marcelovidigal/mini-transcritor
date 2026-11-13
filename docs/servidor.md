# Modo servidor (web) — Fase 1

Versão web **privada** do Mini Transcritor: usuários entram com a **conta Google** (lista de autorizados), **enviam** o arquivo pelo navegador, aguardam numa **fila única** e **baixam** o `.srt`. Tudo o que é temporário é apagado.

## 1. Como escolher entre o modo web e o executável

É o **mesmo código**; o modo é definido pelo **ponto de entrada**:

| Quero… | Como | Modo |
|--------|------|------|
| Usar no meu computador (um usuário, arquivos por caminho, configurações, sem login) | `python app.py` ou `MiniTranscritor.exe` | **local** |
| Publicar para outras pessoas (upload, login Google, fila, arquivos temporários) | `python server.py` ou `docker compose up -d` | **servidor** |
| Gerar o executável (Windows/Linux/macOS) | `python build.py` / `python release.py` / tag no GitHub | só **local** |

O executável **nunca** roda em modo servidor (as bibliotecas do servidor, `authlib` e `waitress`, são excluídas do pacote). O modo servidor não depende de `build.py`.

## 2. Fluxo do usuário

1. Entra em `https://seu-dominio` → **Entrar com Google**. Só e-mails/domínios autorizados passam.
2. Escolhe o arquivo (upload), o idioma e o modelo e clica em **Gerar legenda**. A barra mostra: *Enviando → Na fila (posição N) → etapas do processamento*.
3. Ao concluir aparece **Baixar .srt**. A legenda continua disponível enquanto o usuário **não baixar**, **não clicar em Limpar** e **não escolher outro arquivo**.
4. Ao clicar em **Baixar .srt**, o download começa, a legenda é apagada do servidor e **a tela volta ao estado inicial**.
5. Se escolher outro arquivo (ou *Limpar*) antes de baixar, a legenda guardada é apagada e a tela zera.
6. Recarregar a página no meio do processo **retoma** de onde estava.

Regras: **uma legenda por usuário** por vez (em fila, processando ou aguardando download); fila **única** FIFO para todos (um job processado por vez); a tela só aceita **modelos já instalados** (usuários não disparam downloads de modelos).

## 3. Arquivos temporários

```
<SERVER_DATA_DIR>/upload/<id>.<ext>   ← arquivo enviado. Apagado ao FIM do processamento (sucesso, erro ou cancelamento).
<SERVER_DATA_DIR>/download/<id>.srt   ← legenda pronta. Apagada: ao baixar · ao Limpar/trocar de arquivo · por TTL.
```

- **TTL** (`JOB_TTL_MIN`, padrão 60 min): legendas não baixadas são removidas após esse tempo (varredura a cada minuto).
- Ao iniciar, as duas pastas são **recriadas vazias** (sobras de execuções anteriores).
- Job cancelado na fila tem o upload apagado imediatamente. Job com erro não deixa nada em `download/`.
- Nome do download: `<nome original>.<idioma>.srt`; internamente o arquivo usa o id do job (sem colisões).

## 4. Controle de acesso (Google)

Fluxo OAuth 2.0 / OpenID Connect (biblioteca Authlib). Após o login, o servidor exige `email_verified` e confere o e-mail contra `ALLOWED_EMAILS` e/ou o domínio contra `ALLOWED_DOMAINS`. **Sem nenhuma lista, o servidor não inicia** (falha fechada). Sessão em cookie `HttpOnly`, `SameSite=Lax` e `Secure` quando `BASE_URL` é https; expira em 12 h. Escritas na API exigem o cabeçalho `X-Requested-With: fetch` (defesa contra CSRF). Cada usuário só enxerga **os próprios** jobs.

### Criar as credenciais no Google

1. [Google Cloud Console](https://console.cloud.google.com) → projeto → *APIs e serviços* → *Tela de consentimento OAuth* (tipo **Externo** ou **Interno** para Workspace; em modo *Teste*, adicione os e-mails como usuários de teste).
2. *Credenciais* → *Criar credenciais* → **ID do cliente OAuth** → *Aplicativo da Web*.
3. **URIs de redirecionamento autorizados:** `https://seu-dominio/auth/callback` (igual a `BASE_URL` + `/auth/callback`; para teste local use `http://localhost:8000/auth/callback`).
4. Copie *ID do cliente* e *Chave secreta* para `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET`.

## 4.1 Onde ficam as credenciais: o arquivo `.env`

O `server.py` lê automaticamente o arquivo **`.env`** da pasta do projeto (o Docker Compose também o usa). Variáveis já definidas no ambiente têm prioridade sobre o arquivo. O `.env` **está no `.gitignore`** e **nunca deve ser commitado nem colado em chats/documentos**: ele contém a chave secreta do Google. O modelo sem segredos é o `.env.example` (este sim é versionado). Para um teste local: preencha `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `ALLOWED_EMAILS`, deixe `BASE_URL=http://localhost:8000` (igual à URI cadastrada no Google) e rode `python server.py`; abra `http://localhost:8000` (não `127.0.0.1`).

Se uma chave secreta for exposta, gere outra em *Credenciais → (seu ID de cliente) → Adicionar secret* no Google Cloud Console, atualize o `.env` e desative a antiga.

## 5. Configuração (variáveis de ambiente)

| Variável | Obrigatória | Descrição |
|----------|:-----------:|-----------|
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | sim | credenciais OAuth |
| `SECRET_KEY` | sim | chave da sessão (≥ 16 caracteres) |
| `ALLOWED_EMAILS` / `ALLOWED_DOMAINS` | uma das duas | quem pode entrar (vírgulas) |
| `BASE_URL` | recomendado | URL pública (define o redirect URI e o cookie `Secure`) |
| `OLLAMA_HOST` | — | servidor Ollama (compose: `http://ollama:11434`) |
| `SERVER_DATA_DIR` | — | pasta de `upload/` e `download/` (padrão `server_data`) |
| `MAX_UPLOAD_MB` | — | padrão 500 |
| `MAX_DURATION_MIN` | — | duração máxima da mídia (padrão 180) |
| `JOB_TTL_MIN` | — | validade de legendas não baixadas (padrão 60) |
| `TRUST_PROXY` | — | `1` atrás de Caddy/nginx (respeita `X-Forwarded-*`) |
| `HOST` / `PORT` | — | bind (padrão `0.0.0.0` / `8000`) |

## 6. Publicar com Docker Compose (recomendado)

Requisitos: servidor Linux com Docker e **GPU NVIDIA** + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) (sem GPU, remova o bloco `deploy` do `ollama` — funciona, porém muito lento), domínio apontando para o servidor, portas 80/443 abertas.

```bash
cp .env.example .env      # preencha DOMAIN, BASE_URL, GOOGLE_*, SECRET_KEY, ALLOWED_*
docker compose up -d --build
```

Três serviços: **caddy** (HTTPS automático, único com portas publicadas) → **app** (waitress, 1 processo/8 threads — o estado da fila fica na memória) → **ollama** (rede interna, **nunca exposto**). Na primeira subida o app baixa o modelo base (`gemma4:e4b`) automaticamente; acompanhe pela tela ("Preparando o ambiente") ou `docker compose logs -f app`. Para outros modelos: `docker compose exec ollama ollama pull <modelo>:<tag>` com a tag descritiva (ex.: `gemma4:e4b`, nunca `latest`).

Sem Docker: `pip install -r requirements.txt`, exporte as variáveis da seção 5 e rode `python server.py` atrás do seu proxy HTTPS (Ollama e FFmpeg instalados no servidor).

## 7. Segurança e operação

- Nunca exponha o Ollama (porta 11434) à internet; nunca commite o `.env`.
- Sempre use HTTPS (sessão e arquivos de áudio trafegam pela rede).
- Os áudios dos usuários passam pelo seu servidor: informe isso a eles; os arquivos existem só durante o processamento.
- Sem limite de requisições por IP nesta fase (apenas: 1 job por usuário, limite de tamanho e de duração). Se expor a mais gente, adicione limitação no Caddy/nginx.
- `/healthz` é público e só devolve `ok` e a versão.
- Backup: não há dados a preservar (tudo é temporário); só os modelos do Ollama (volume `ollama_models`).

## 8. Estado de verificação

- **Testado aqui:** 23 testes automatizados do modo servidor (login com dublê do Google, lista de autorizados, CSRF, isolamento entre usuários, ciclo de vida de `upload/` e `download/`, fila FIFO e cancelamento na fila, 1 job por usuário, limites de tamanho/duração, TTL, limpeza na inicialização); `python server.py` subindo com waitress e recusando iniciar sem configuração; fluxo completo com Ollama e FFmpeg reais (upload → fila → transcrição → download → tudo apagado).
- **NÃO testado:** o login com o Google **real** (exige suas credenciais e domínio — o fluxo foi testado com dublê); a construção da imagem Docker e a subida do Compose em servidor com GPU (o `docker compose config` valida; o daemon não estava disponível aqui); HTTPS via Caddy; a interface no navegador (o JS foi checado por sintaxe e pela API, não por navegação).
- Antes de abrir para outras pessoas, faça um teste com sua própria conta e outra **não autorizada** (deve ver "não tem acesso").

## 9. Limitações da Fase 1 e próximos passos

Estado em memória (reiniciar o contêiner perde jobs em andamento; os arquivos são recriados vazios) · um único worker de transcrição · sem contas/quotas próprias (acesso por lista) · sem armazenamento de objetos. Fase 2 (público): banco de dados e fila persistente (Redis/RQ), S3 para arquivos, quotas, observabilidade.
