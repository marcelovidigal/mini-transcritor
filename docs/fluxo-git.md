# Fluxo de branches (Git)

Duas branches fixas, sem burocracia extra:

| Branch | Papel | Quem escreve nela |
|---|---|---|
| `dev` | desenvolvimento do dia a dia; pode estar "em obras", mas a CI deve ficar verde | commits diretos |
| `main` | sempre a versão que funciona; é de onde saem os releases | **só** por pull request vindo de `dev` |

## Dia a dia

1. Trabalhe e faça commits na `dev` (`git switch dev`). A cada push a CI roda nos três sistemas (Windows, Linux e macOS).
2. Quando a `dev` estiver estável e a CI verde, abra um **pull request `dev` → `main`** no GitHub.
3. Com a CI do pull request verde, faça o merge (**squash**: um commit por entrega na `main`, com a descrição do que mudou).
4. Para voltar a trabalhar: `git switch dev && git pull`. Depois de um squash, alinhe a `dev` com a `main` (`git fetch && git reset --hard origin/main` na `dev`, só se não houver nada novo nela) para os históricos não divergirem.

## Release

O release é automático e acontece **no merge `dev` → `main`, quando a versão em `transcritor/version.py` é nova** (ainda sem Release/tag `vX.Y.Z`): o pipeline compila os executáveis, publica o Release e a imagem Docker (GitHub Packages). Merge com a mesma versão não publica nada. Passo a passo, a execução manual sem publicar e o caminho por tag: [release.md](release.md).

## Regras da `main` (GitHub → Settings → Branches → Add rule, padrão `main`)

- **Require a pull request before merging** (sem commits diretos).
- **Require status checks to pass before merging**, marcando os três jobs da CI: `test (windows-latest)`, `test (ubuntu-latest)` e `test (macos-latest)`.
- **Require branches to be up to date before merging** (a CI testa o resultado do merge).
- Opcional: **Allow squash merging** apenas (Settings → General → Pull Requests) e **Automatically delete head branches** desligado, para a `dev` não ser apagada após o merge.

Os checks só aparecem na lista depois que a CI rodou ao menos uma vez com esses nomes.

## Mudanças grandes ou arriscadas

Crie uma branch curta a partir da `dev` (`feat/...`), faça o pull request para a `dev` e junte quando estiver pronta. O fluxo normal não exige isso.
