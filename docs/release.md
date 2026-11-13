# Release e distribuição multiplataforma

## 1. Resposta curta

- **Dá para gerar executáveis para Windows, Linux x64, Linux arm64 e macOS**, mas o PyInstaller **não faz cross-compile**: cada plataforma precisa ser compilada no próprio sistema. A solução é uma **matriz de runners nativos no GitHub Actions** (`.github/workflows/release.yml`).
- **Releases:** o modelo do GitHub é **um release por versão** (tag `vX.Y.Z`) com **um arquivo por plataforma** anexado. É o recomendado: uma única página de download, notas de versão únicas, um só número de versão. Três releases separados (um por SO) exigiriam tags diferentes (`v1.0.0-windows`…), duplicariam notas e confundiriam a versão; só valeria a pena se as plataformas tivessem ciclos de vida independentes.

Artefatos por release:

| Arquivo | Plataforma | Runner |
|---------|-----------|--------|
| `MiniTranscritor-vX.Y.Z-windows-x64.zip` | Windows 64 bits | `windows-latest` |
| `MiniTranscritor-vX.Y.Z-linux-x64.tar.gz` | Linux x86-64 | `ubuntu-latest` |
| `MiniTranscritor-vX.Y.Z-linux-arm64.tar.gz` | Linux ARM64 | `ubuntu-24.04-arm` |
| `MiniTranscritor-vX.Y.Z-macos-arm64.tar.gz` | macOS Apple Silicon | `macos-latest` |
| `MiniTranscritor-vX.Y.Z-macos-x64.tar.gz` | macOS com processador Intel | `macos-15-intel` |

Cada arquivo acompanha um `.sha256`. O rótulo `macos-15-intel` do runner Intel não foi conferido por execução: se o GitHub o aposentar ou renomear, o job do macOS Intel falha e a linha da matriz em `release.yml` precisa ser ajustada. Como a matriz tem `fail-fast: false`, os outros sistemas continuam, mas a publicação só ocorre se **todos** passarem.

## 2. Estratégia: o merge na `main` lança, mas só quando a versão é nova

O workflow `release.yml` é a **entrega contínua (CD)**. Ele roda a cada push na `main` (o merge do pull request `dev` → `main`), mas **só publica se a versão de `transcritor/version.py` ainda não tem Release**. Merge comum (mesma versão) não compila nem publica nada. Quando publica, faz duas coisas: o **Release** do GitHub com os 4 executáveis e a **imagem Docker** do servidor em `ghcr.io/marcelovidigal/mini-transcritor:<versão>` e `:latest` (pré-releases, versão com `-`, não recebem `:latest`).


| Situação | O que acontece |
|----------|----------------|
| `git push` em `dev`/`main` ou PR para a `main` | **CI** (`ci.yml`): roda os testes em Windows, Linux e macOS. **Não gera release.** |
| Merge na `main` com versão **nova** em `version.py` | **CD** (`release.yml`): testa, compila as 5 plataformas, publica o Release (a tag `vX.Y.Z` é criada junto) e a imagem Docker. |
| Merge na `main` com a **mesma** versão | Nada é compilado nem publicado (o job "Decidir o que fazer" avisa e encerra). |
| Execução manual do workflow *Release* com `publish` desmarcado | Gera os 5 artefatos (baixáveis na aba *Actions*) e **compila a imagem Docker sem enviá-la** (valida o `Dockerfile`), para **conferência**. Nada é publicado. |
| Execução manual com `publish` marcado | Gera e **publica** o release e a imagem `v<versão de version.py>`; falha se esse Release já existe. |
| Push da tag `vX.Y.Z` | Valida que a tag = `transcritor/version.py`, testa, compila as 5 plataformas e **publica**. Tags com `-` (ex.: `v1.1.0-rc1`) viram *pre-release*. |

Cada compilação executa, nesta ordem: **testes → build do executável → smoke test do executável → artefato (+ sha256)**. A publicação só ocorre se todas as plataformas passaram.

### Passo a passo de um release

1. Na `dev`, altere `transcritor/version.py` (fonte única da versão) e atualize notas/docs.
2. (Opcional) Ensaio local na sua plataforma: `python release.py` — roda testes, build, smoke test e gera o arquivo em `dist/`, sem publicar.
3. Confira antes de lançar: *Actions → Release → Run workflow* na `dev`/`main` com `publish` desmarcado (gera os 5 artefatos para baixar e testar).
4. Abra o pull request `dev` → `main`; com a CI verde, faça o merge (squash). **O merge publica** o Release e a imagem Docker.
5. Caminho alternativo manual: `git tag v1.1.0 && git push origin v1.1.0` (a tag precisa bater com `version.py`; se o Release já existir, nada é feito).

`python release.py --publish` publica pelo `gh` **somente o artefato do seu SO**; use-o apenas para releases só-Windows ou emergências. O release completo vem do CI.

## 3. O que cada plataforma exige do usuário

| | Windows | Linux | macOS |
|---|---------|-------|-------|
| Ollama | instalador oficial (o app baixa e abre) | `curl -fsSL https://ollama.com/install.sh \| sh` | app em ollama.com/download |
| FFmpeg | baixado automaticamente para `ffmpeg/` | `apt install ffmpeg` (ou equivalente) | `brew install ffmpeg` |
| Aviso de segurança | SmartScreen (binário não assinado) | — | Gatekeeper: `xattr -dr com.apple.quarantine MiniTranscritor/` |

## 4. Estado de verificação (honestidade)

- **Testado de verdade (Windows x64):** testes (42), build, smoke test do `.exe`, `release.py` em dry-run, download local do FFmpeg, início do Ollama pelo app.
- **Escrito, mas NÃO executado aqui:** os workflows do GitHub Actions (YAML validado sintaticamente), e as builds Linux x64/arm64 e macOS. Rode o workflow manual com `publish` desmarcado antes do primeiro release real e corrija o que surgir.
- Código específico de plataforma (instalador do Ollama, download do FFmpeg) só roda no Windows; nos demais, o app mostra como instalar.
- O smoke test usa `MINITRANSCRITOR_NO_BOOTSTRAP=1` (sem Ollama/FFmpeg no CI); a transcrição real não é exercitada em CI.

## 5. Evoluções possíveis

Assinatura de código (Windows) e assinatura + notarização (macOS); instalador (`.msi`/`.dmg`); download automático do FFmpeg em Linux/macOS.
