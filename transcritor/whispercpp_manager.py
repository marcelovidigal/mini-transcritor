"""Download e instalação local dos executáveis do whisper.cpp (`whisper-cli`), em duas variantes: CPU e GPU NVIDIA (CUDA).

Os binários vêm dos releases oficiais (github.com/ggml-org/whisper.cpp/releases) e ficam em `<app>/whispercpp/<variante>/`,
lado a lado (duas versões baixadas, escolhidas pelo motor). Só há binário CUDA pronto para Windows x64; para Intel/AMD
(Vulkan, SYCL) é preciso compilar. Nomes dos assets variaram entre releases, por isso há vários padrões por variante.
"""
import hashlib
import os
import platform
import re
import shutil
import sys
import tarfile
import tempfile
import zipfile

from .ffmpeg_manager import app_dir

API = "https://api.github.com/repos/ggml-org/whisper.cpp/releases?per_page=30"
CPU, CUDA = "cpu", "cuda"
# ordem de preferência das versões do CUDA: a 12.x é a mais compatível com drivers atuais
CUDA_PREFERENCE = (12, 13, 11)


class ManagerError(Exception):
    pass


def machine() -> tuple[str, str]:
    os_name = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    arch = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(platform.machine().lower(), platform.machine().lower())
    return os_name, arch


def _patterns(variant: str, os_name: str, arch: str) -> list[re.Pattern]:
    if variant == CPU:
        names = {("windows", "x64"): [r"whisper-bin-x64\.zip", r"whisper-bin-win-cpu-x64\.zip"],
                 ("windows", "arm64"): [r"whisper-bin-win-cpu-arm64\.zip"],
                 ("linux", "x64"): [r"whisper-bin-ubuntu-x64\.tar\.gz"],
                 ("linux", "arm64"): [r"whisper-bin-ubuntu-arm64\.tar\.gz"]}.get((os_name, arch), [])
    elif variant == CUDA:
        names = {("windows", "x64"): [r"whisper-cublas-[\d.]+-bin-x64\.zip", r"whisper-bin-win-cuda-[\d.]+-x64\.zip"]}.get((os_name, arch), [])
    else:
        names = []
    return [re.compile("^" + n + "$") for n in names]


def supported(variant: str) -> bool:
    return bool(_patterns(variant, *machine()))


def _cuda_version(name: str) -> tuple[int, ...]:
    m = re.search(r"(\d+(?:\.\d+)*)", name.replace("whisper-bin-win-cuda-", "").replace("whisper-cublas-", ""))
    return tuple(int(x) for x in m.group(1).split(".")) if m else ()


def pick_asset(assets: list[dict], variant: str, os_name: str | None = None, arch: str | None = None) -> dict | None:
    """Escolhe, entre os assets de UM release, o da variante pedida (ou None). Para CUDA prefere a série 12.x."""
    if os_name is None or arch is None:
        os_name, arch = machine()
    pats = _patterns(variant, os_name, arch)
    found = [a for a in assets if any(p.match(a.get("name", "")) for p in pats)]
    if not found:
        return None
    if variant == CUDA:
        def rank(a):
            v = _cuda_version(a["name"])
            major = v[0] if v else 0
            return (CUDA_PREFERENCE.index(major) if major in CUDA_PREFERENCE else len(CUDA_PREFERENCE), tuple(-x for x in v))
        return sorted(found, key=rank)[0]
    return found[0]


def find_release_asset(variant: str, http=None) -> dict:
    """Primeiro release (do mais novo ao mais antigo) que tem o binário da variante. Levanta ManagerError se não houver."""
    import requests

    http = http or requests
    if not supported(variant):
        raise ManagerError(f"Não há binário pronto do whisper.cpp ({variant}) para {'/'.join(machine())}: compile a partir do código-fonte.")
    try:
        r = http.get(API, timeout=30, headers={"Accept": "application/vnd.github+json"})
        r.raise_for_status()
        releases = r.json()
    except Exception as e:
        raise ManagerError(f"Não foi possível consultar os releases do whisper.cpp no GitHub: {e}")
    for rel in releases:
        asset = pick_asset(rel.get("assets", []), variant)
        if asset:
            return {"name": asset["name"], "url": asset["browser_download_url"], "size": asset.get("size", 0),
                    "digest": asset.get("digest") or "", "tag": rel.get("tag_name", "")}
    raise ManagerError(f"Nenhum release do whisper.cpp tem o binário '{variant}' para {'/'.join(machine())}.")


def variant_dir(variant: str) -> str:
    return os.path.join(app_dir(), "whispercpp", variant)


def _find_cli(root: str) -> str | None:
    names = ("whisper-cli.exe", "whisper-cli")
    for dirpath, _dirs, files in os.walk(root):
        for n in names:
            if n in files:
                return os.path.join(dirpath, n)
    return None


def install(variant: str, progress=None, http=None, dest: str | None = None) -> str:
    """Baixa o release da variante e instala em `<app>/whispercpp/<variante>/`. Devolve o caminho do whisper-cli."""
    import requests

    http = http or requests
    asset = find_release_asset(variant, http)
    dest = dest or variant_dir(variant)
    tmp = tempfile.mkdtemp(prefix="mt_wcpp_dl_")
    try:
        archive = os.path.join(tmp, asset["name"])
        sha = hashlib.sha256()
        try:
            with http.get(asset["url"], stream=True, timeout=(15, 120), allow_redirects=True) as r:
                r.raise_for_status()
                total, done = int(r.headers.get("content-length", asset.get("size", 0) or 0)), 0
                with open(archive, "wb") as f:
                    for block in r.iter_content(1 << 20):
                        f.write(block)
                        sha.update(block)
                        done += len(block)
                        if progress:
                            progress(done, total)
        except Exception as e:
            raise ManagerError(f"Falha ao baixar {asset['name']}: {e}")
        expected = asset["digest"].split(":", 1)[-1].lower() if asset["digest"].lower().startswith("sha256:") else ""
        if expected and sha.hexdigest() != expected:
            raise ManagerError(f"O download de {asset['name']} está corrompido (SHA-256 não confere). Tente novamente.")
        out = os.path.join(tmp, "x")
        os.makedirs(out)
        if archive.endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                z.extractall(out)
        else:
            with tarfile.open(archive) as t:
                t.extractall(out)
        cli = _find_cli(out)
        if not cli:
            raise ManagerError(f"O pacote {asset['name']} não contém o whisper-cli.")
        root = os.path.dirname(cli)  # a pasta do executável carrega as DLLs/bibliotecas junto
        if os.path.isdir(dest):
            shutil.rmtree(dest, ignore_errors=True)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copytree(root, dest)
        final = os.path.join(dest, os.path.basename(cli))
        if sys.platform != "win32":
            os.chmod(final, 0o755)
        return final
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def has_nvidia() -> bool:
    """Há uma GPU NVIDIA utilizável? (driver instalado: nvidia-smi no PATH ou nvcuda.dll no sistema)."""
    if shutil.which("nvidia-smi"):
        return True
    if sys.platform == "win32":
        return os.path.isfile(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvcuda.dll"))
    return os.path.exists("/proc/driver/nvidia/version")
