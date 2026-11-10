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

Cada arquivo acompanha um `.sha256`. macOS Intel não está na matriz (há uma linha comentada em `release.yml`; confirme o rótulo do runner Intel disponível antes de ativar).

## 2. Estratégia: você decide quando há release

| Situação | O que acontece |
|----------|----------------|
| `git push` / PR | **CI** (`ci.yml`): roda os testes em Windows, Linux e macOS. **Não gera release.** |
| Execução manual do workflow *Release* com `publish` desmarcado | Gera os 4 artefatos (baixáveis na aba *Actions*) para **conferência**. Nada é publicado. |
| Execução manual com `publish` marcado | Gera e **publica** o release `v<versão de version.py>`. |
| Push da tag `vX.Y.Z` | Valida que a tag = `transcritor/version.py`, testa, compila as 4 plataformas e **publica**. Tags com `-` (ex.: `v1.1.0-rc1`) viram *pre-release*. |

Cada compilação executa, nesta ordem: **testes → build do executável → smoke test do executável → artefato (+ sha256)**. A publicação só ocorre se todas as plataformas passaram.

### Passo a passo de um release

1. Altere `transcritor/version.py` (fonte única da versão) e atualize notas/docs.
2. (Opcional) Ensaio local na sua plataforma: `python release.py` — roda testes, build, smoke test e gera o arquivo em `dist/`, sem publicar.
3. Commit e push em `main`; aguarde o CI ficar verde.
4. Publique: `git tag v1.1.0 && git push origin v1.1.0`. O workflow cria o release com os 4 arquivos e notas geradas automaticamente.
5. Se quiser apenas conferir antes: *Actions → Release → Run workflow* com `publish` desmarcado.

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

Assinatura de código (Windows) e assinatura + notarização (macOS); instalador (`.msi`/`.dmg`); download automático do FFmpeg em Linux/macOS; build macOS Intel.
