import os
import time

from transcritor import catalog

from .test_api import start, wait_done


def test_parse_ignores_comments_and_duplicates():
    items = catalog.parse("# c\n\n a:1 | 1 GB | nota \nA:1 | x\nsem espaco:2 | y\nb:2b-q4_K_M | ?\nsemtag | z\nm:latest | z\n")
    assert [i["name"] for i in items] == ["a:1", "b:2b-q4_K_M"]  # sem tag / `latest` são ignorados
    assert items[0] == {"name": "a:1", "size": "1 GB", "note": "nota"} and items[1]["size"] == "?"


def test_shipped_catalog_uses_only_explicit_descriptive_tags():
    raw = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio_models.txt"),
               encoding="utf-8").read()
    names = [l.split("|")[0].strip() for l in raw.splitlines() if l.strip() and not l.startswith(("#", "!"))]
    assert names and all(catalog.has_explicit_tag(n) for n in names), names  # nada de latest/sem tag
    assert names == [i["name"] for i in catalog.load()]                      # e o parser não descartou nenhum


def test_settings_base_model_requires_explicit_tag(tmp_path):
    import pytest
    from transcritor.settings import Settings, SettingsError

    s = Settings(str(tmp_path / "s.json"))
    for bad in ("gemma4", "gemma4:latest", "Gemma4:LATEST"):
        with pytest.raises(SettingsError):
            s.update({"base_model": bad})
    assert s.update({"base_model": "nemotron3:33b-q4_K_M"})["base_model"] == "nemotron3:33b-q4_K_M"


def test_shipped_catalog_is_gemma_first_and_valid():
    items = catalog.load()
    assert items[0]["name"] == "gemma4:e4b" and [i["name"] for i in items[:3]] == ["gemma4:e4b", "gemma4:e2b", "gemma4:12b"]
    assert all(" " not in i["name"] for i in items)


def test_merge_marks_installed_and_orders():
    cat = [{"name": "gemma4:e4b", "size": "6.6 GB", "note": "n"}, {"name": "gemma4:e2b", "size": "4.6 GB", "note": ""},
           {"name": "nemotron3", "size": "", "note": ""}]
    inst = [{"name": "gemma4:e2b", "size": 5}, {"name": "nemotron3:latest", "size": 9}, {"name": "meu:audio", "size": 1}]
    m = catalog.merge(cat, inst)
    assert [(x["name"], x["installed"]) for x in m] == [
        ("gemma4:e2b", True), ("nemotron3:latest", True), ("meu:audio", True), ("gemma4:e4b", False)]
    assert m[2]["recommended"] is False and m[3]["recommended"] is True


def test_models_api_lists_catalog_with_installed_first(client_factory):
    c, _ = client_factory()
    models = c.get("/api/models").get_json()["models"]
    assert models[0] == {**models[0], "name": "fake-audio:1b", "installed": True, "recommended": False}
    names = [m["name"] for m in models]
    assert names.index("gemma4:e4b") < names.index("gemma4:e2b")
    assert not next(m for m in models if m["name"] == "gemma4:e4b")["installed"]


def test_job_with_catalog_model_not_installed_pulls_first(client_factory, media_dir, tmp_path):
    import shutil

    src = tmp_path / "x.mp3"
    shutil.copy(media_dir / "a.mp3", src)
    c, fake = client_factory()
    r = start(c, src, model="gemma4:e2b")
    assert r.status_code == 202
    s, seen = wait_done(c, r.get_json()["id"])
    assert s["status"] == "done", s
    assert fake.pulled == ["gemma4:e2b"] and seen == sorted(seen) and s["progress"] == 100
    assert {call[0] for call in fake.calls} == {"gemma4:e2b"}


def test_installed_model_is_not_pulled(client_factory, media_dir, tmp_path):
    import shutil

    src = tmp_path / "y.mp3"
    shutil.copy(media_dir / "a.mp3", src)
    c, fake = client_factory()
    s, _ = wait_done(c, start(c, src).get_json()["id"])
    assert s["status"] == "done" and not getattr(fake, "pulled", [])


def test_excluded_models_disappear_from_options_even_when_installed():
    text = "gemma4:e4b | 6.6 GB | ok\n!lfm3-omni:2b-q4_K_M | fraco em português\n"
    assert catalog.parse_excluded(text) == {"lfm3-omni:2b-q4_k_m"}
    assert [m["name"] for m in catalog.parse(text)] == ["gemma4:e4b"]  # a linha com ! não vira modelo
    installed = [{"name": "lfm3-omni:2b-q4_K_M", "size": 2}, {"name": "gemma4:e4b", "size": 6}]
    names = [m["name"] for m in catalog.merge(catalog.parse(text), installed, catalog.parse_excluded(text))]
    assert names == ["gemma4:e4b"]
    assert "lfm3-omni:2b-q4_K_M" in [m["name"] for m in catalog.merge(catalog.parse(text), installed)]  # sem a lista, aparece
