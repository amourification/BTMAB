from pathlib import Path

from gui.theme import get_current_theme, persist_theme_choice
import config as app_config


def test_get_current_theme_returns_known_keys():
    theme = get_current_theme()
    for key in ("BG", "PANEL", "BORDER", "FG", "ACCENT"):
        assert key in theme
        assert isinstance(theme[key], str) and theme[key]


def test_persist_theme_choice_writes_env_and_updates_config(tmp_path, monkeypatch):
    # Point BASE_DIR to a temporary folder so we don't touch the real .env
    fake_base = tmp_path
    env_path = fake_base / ".env"
    env_path.write_text("BINANCE_API_KEY=dummy\n", encoding="utf-8")

    monkeypatch.setattr(app_config, "BASE_DIR", fake_base, raising=False)
    # Reset GUI_THEME to something neutral
    app_config.GUI_THEME = "dark"

    persist_theme_choice("light")

    text = env_path.read_text(encoding="utf-8")
    assert "GUI_THEME=light" in text
    assert app_config.GUI_THEME == "light"

