"""Última pasta de trabalho: o navegador de arquivos abre onde o usuário parou (e cai no padrão se ela sumir)."""
import os


def test_browse_starts_in_the_last_folder_and_falls_back(client_factory, tmp_path):
    c, _ = client_factory()
    work = tmp_path / "trabalho"
    work.mkdir()
    (work / "aula.mp3").write_bytes(b"x")
    assert c.get(f"/api/browse?path={work}").get_json()["ok"]  # abrir a pasta a memoriza
    r = c.get("/api/browse?start=1").get_json()
    assert r["ok"] and r["path"] == str(work) and [f["name"] for f in r["files"]] == ["aula.mp3"]
    assert c.get("/api/settings").get_json()["last_dir"] == str(work)
    # navegar para cima não é abertura do diálogo: respeita o caminho pedido (inclusive a raiz de unidades)
    assert c.get("/api/browse?path=").get_json()["path"] != str(work)
    # a pasta pedida (do campo) vence a última; sumiu: volta ao padrão
    other = tmp_path / "outra"
    other.mkdir()
    assert c.get(f"/api/browse?start=1&path={other}").get_json()["path"] == str(other)
    work_str = str(other)
    os.rmdir(other)
    r = c.get("/api/browse?start=1").get_json()
    assert r["ok"] and r["path"] != work_str


def test_starting_a_job_remembers_the_media_folder(client_factory, tmp_path):
    c, _ = client_factory()
    media = tmp_path / "pasta_midia"
    media.mkdir()
    f = media / "x.mp3"
    f.write_bytes(b"x")
    c.post("/api/jobs", json={"path": str(f), "language": "pt-BR", "model": "m"})  # o resultado do job não importa aqui
    assert c.get("/api/settings").get_json()["last_dir"] == str(media)
