import os

from transcritor import envfile


def test_parse_handles_comments_quotes_and_blank_values():
    d = envfile.parse('# c\n\nA=1\nB = "x y"\nC=\'z\'\nD=val # comentário\nE=\nSEM_IGUAL\n=sem_chave\n')
    assert d == {"A": "1", "B": "x y", "C": "z", "D": "val", "E": ""}


def test_parse_keeps_equals_in_value():
    assert envfile.parse("K=abc=def==")["K"] == "abc=def=="


def test_load_does_not_override_existing_env_and_never_returns_values(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("MT_A=do_arquivo\nMT_B=novo\n", encoding="utf-8")
    monkeypatch.setenv("MT_A", "do_ambiente")
    monkeypatch.delenv("MT_B", raising=False)
    loaded = envfile.load(str(f))
    assert loaded == ["MT_B"] and os.environ["MT_A"] == "do_ambiente" and os.environ["MT_B"] == "novo"
    os.environ.pop("MT_B", None)


def test_load_missing_file_is_noop(tmp_path):
    assert envfile.load(str(tmp_path / "nao_existe")) == []
