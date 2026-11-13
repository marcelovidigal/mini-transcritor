"""Carrega um arquivo .env (CHAVE=valor) para os.environ, sem dependências externas.

Variáveis já definidas no ambiente têm prioridade (não são sobrescritas)."""
import os


def parse(text: str) -> dict:
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:  # comentário no fim da linha
            value = value.split(" #", 1)[0].rstrip()
        if key:
            out[key] = value
    return out


def load(path: str) -> list[str]:
    """Aplica o .env em os.environ; devolve os nomes carregados (nunca os valores)."""
    try:
        with open(path, encoding="utf-8") as f:
            data = parse(f.read())
    except OSError:
        return []
    loaded = []
    for k, v in data.items():
        if k not in os.environ:
            os.environ[k] = v
            loaded.append(k)
    return loaded
