"""Download e instalação local do NeMo-Speech.cpp (`nemo-speech`), o runtime C++ oficial da NVIDIA para os modelos Nemotron.

Os binários vêm dos releases oficiais (github.com/NVIDIA/NeMo-Speech.cpp/releases) e ficam em `<app>/nemocpp/<backend>/`.
Há um pacote por sistema e backend: `nemo-speech-<versão>-<so>-<arquitetura>-<backend>.zip|.tar.gz`, com backend
`cuda` (GPU NVIDIA), `vulkan` (qualquer GPU com driver Vulkan: Intel Arc, AMD, NVIDIA), `metal` (Apple Silicon) ou `cpu`.
O backend é escolhido pela máquina: NVIDIA → cuda; outra GPU → vulkan; senão cpu. Só um é instalado por vez.
"""
import hashlib
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import zipfile

from .ffmpeg_manager import app_dir

API = "https://api.github.com/repos/NVIDIA/NeMo-Speech.cpp/releases?per_page=10"
BACKENDS = ("cuda", "vulkan", "metal", "cpu")


class ManagerError(Exception):
    pass


def machine() -> tuple[str, str]:
    os_name = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}.get(platform.machine().lower(), platform.machine().lower())
    return os_name, arch


def has_nvidia() -> bool:
    if shutil.which("nvidia-smi"):
        return True
    if sys.platform == "win32":
        return os.path.isfile(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvcuda.dll"))
    return os.path.exists("/proc/driver/nvidia/version")


def has_vulkan() -> bool:
    """Há um carregador Vulkan no sistema? (vulkan-1.dll no Windows; libvulkan no Linux)."""
    if sys.platform == "win32":
        return os.path.isfile(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "vulkan-1.dll"))
    if sys.platform == "darwin":
        return False
    return any(os.path.exists(p) for p in ("/usr/lib/x86_64-linux-gnu/libvulkan.so.1", "/usr/lib/aarch64-linux-gnu/libvulkan.so.1",
                                            "/usr/lib64/libvulkan.so.1", "/usr/lib/libvulkan.so.1"))


def preferred_backends(os_name: str | None = None, arch: str | None = None, nvidia: bool | None = None, vulkan: bool | None = None) -> list[str]:
    """Backends a tentar, do melhor ao pior para esta máquina (o último é sempre a CPU)."""
    if os_name is None or arch is None:
        os_name, arch = machine()
    nvidia = has_nvidia() if nvidia is None else nvidia
    vulkan = has_vulkan() if vulkan is None else vulkan
    if os_name == "macos":
        return ["metal", "cpu"] if arch == "aarch64" else ["cpu"]
    out = []
    if nvidia and os_name in ("windows", "linux") and arch in ("x86_64", "aarch64"):
        out.append("cuda")
    if vulkan or not nvidia:
        out.append("vulkan")
    out.append("cpu")
    return out


def asset_name_matches(name: str, os_name: str, arch: str, backend: str) -> bool:
    """`nemo-speech-<versão>-<so>-<arq>-<backend>.zip|.tar.gz` (a versão pode ser 0.3.0 ou nightly); ignora os .sha256."""
    if not name.startswith("nemo-speech-") or not name.endswith((".zip", ".tar.gz")):
        return False
    stem = name[: -len(".zip")] if name.endswith(".zip") else name[: -len(".tar.gz")]
    return stem.endswith(f"-{os_name}-{arch}-{backend}") or (backend == "cuda" and f"-{os_name}-{arch}-cuda" in stem)


def pick_asset(assets: list[dict], backend: str, os_name: str | None = None, arch: str | None = None) -> dict | None:
    if os_name is None or arch is None:
        os_name, arch = machine()
    for a in assets:
        if asset_name_matches(a.get("name", ""), os_name, arch, backend):
            return a
    return None


def find_release_asset(backends: list[str] | None = None, http=None) -> dict:
    """Primeiro release ESTÁVEL (do mais novo ao mais antigo) com pacote para a máquina; tenta cada backend em ordem de preferência."""
    import requests

    http = http or requests
    backends = backends or preferred_backends()
    try:
        r = http.get(API, timeout=30, headers={"Accept": "application/vnd.github+json"})
        r.raise_for_status()
        releases = [x for x in r.json() if not x.get("prerelease") and not x.get("draft")]
    except Exception as e:
        raise ManagerError(f"Não foi possível consultar os releases do NeMo-Speech.cpp no GitHub: {e}")
    for rel in releases:
        for backend in backends:
            asset = pick_asset(rel.get("assets", []), backend)
            if asset:
                return {"name": asset["name"], "url": asset["browser_download_url"], "size": asset.get("size", 0),
                        "digest": asset.get("digest") or "", "tag": rel.get("tag_name", ""), "backend": backend}
    raise ManagerError("Nenhum release do NeMo-Speech.cpp tem pacote para %s (%s)." % ("/".join(machine()), ", ".join(backends)))


def install_dir() -> str:
    return os.path.join(app_dir(), "nemocpp")


def _find_cli(root: str) -> str | None:
    names = ("nemo-speech.exe", "nemo-speech")
    for dirpath, _dirs, files in os.walk(root):
        for n in names:
            if n in files:
                return os.path.join(dirpath, n)
    return None


def locate(path: str) -> str | None:
    """Acha o `nemo-speech` em `path` (executável, ou pasta com bin/ ou o pacote inteiro dentro)."""
    if not path:
        return None
    if os.path.isfile(path):
        return path
    return _find_cli(path) if os.path.isdir(path) else None


def installed_cli() -> str | None:
    return locate(install_dir())


def install(progress=None, http=None, backends: list[str] | None = None, dest: str | None = None) -> str:
    """Baixa o release e instala em `<app>/nemocpp/`. Devolve o caminho do `nemo-speech`. `progress(baixado, total)` em bytes."""
    import requests

    http = http or requests
    asset = find_release_asset(backends, http)
    dest = dest or install_dir()
    tmp = tempfile.mkdtemp(prefix="mt_nemocpp_dl_")
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
            raise ManagerError(f"O pacote {asset['name']} não contém o nemo-speech.")
        bindir = os.path.dirname(cli)
        root = os.path.dirname(bindir) if os.path.basename(bindir).lower() == "bin" else bindir  # leva junto share/, lib/ e include/
        if os.path.isdir(dest):
            shutil.rmtree(dest, ignore_errors=True)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copytree(root, dest)
        final = _find_cli(dest)
        if sys.platform != "win32" and final:
            os.chmod(final, 0o755)
        with open(os.path.join(dest, "BACKEND.txt"), "w", encoding="utf-8") as f:
            f.write(f"{asset['backend']}\n{asset['tag']}\n")
        return final
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def installed_backend() -> str:
    """Backend do binário instalado ('' se desconhecido)."""
    try:
        with open(os.path.join(install_dir(), "BACKEND.txt"), encoding="utf-8") as f:
            return f.readline().strip()
    except OSError:
        return ""
