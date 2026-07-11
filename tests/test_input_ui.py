import stat

from ghostwheel.input_ui import InputHistory, default_history_path


def test_persistent_input_history_is_private_and_loads_multiline_entries(
    tmp_path,
) -> None:
    history_path = tmp_path / "state" / "history"
    history = InputHistory(history_path)

    history.append("first")
    history.append("more\nthan one line")

    assert stat.S_IMODE(history_path.stat().st_mode) == 0o600
    assert InputHistory(history_path).entries == ["first", "more\nthan one line"]
    rendered = history_path.read_text()
    assert "+first" in rendered
    assert "+more\n+than one line" in rendered


def test_in_memory_history_does_not_create_a_file() -> None:
    history = InputHistory(None)
    history.append("remember for this run")

    assert history.entries == ["remember for this run"]
    assert history.path is None


def test_default_history_path_honors_xdg_state_home(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert default_history_path() == tmp_path / "ghostwheel" / "input-history"
