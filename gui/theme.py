from __future__ import annotations

from pathlib import Path
from typing import Dict

import config as app_config


DARK_THEME: Dict[str, str] = {
    # Inspired by modern Ubuntu dark Yaru theme
    "BG": "#2c001e",       # Aubergine background
    "PANEL": "#3a0b2b",    # Slightly lighter panels
    "BORDER": "#4e1f3d",   # Subtle borders
    "FG": "#f2f1f0",       # Light text
    "ACCENT": "#e95420",   # Ubuntu orange
}

LIGHT_THEME: Dict[str, str] = {
    "BG": "#f2f1f0",       # Ubuntu light background
    "PANEL": "#ffffff",
    "BORDER": "#d3d0cb",
    "FG": "#2c001e",       # Aubergine text
    "ACCENT": "#e95420",   # Ubuntu orange
}

THEMES: Dict[str, Dict[str, str]] = {
    "dark": DARK_THEME,
    "light": LIGHT_THEME,
}


def get_current_theme() -> Dict[str, str]:
    """Returns the theme dict based on GUI_THEME in config."""
    name = getattr(app_config, "GUI_THEME", "dark").lower()
    return THEMES.get(name, DARK_THEME)


def persist_theme_choice(theme_name: str) -> None:
    """
    Persist the chosen theme to .env so it becomes the default on next launch.
    This is a simple overwrite/append of GUI_THEME in the .env file.
    """
    theme_name = theme_name.lower()
    if theme_name not in THEMES:
        return

    base_dir = app_config.BASE_DIR
    env_path = base_dir / ".env"
    lines = []

    if env_path.exists():
        text = env_path.read_text(encoding="utf-8")
        found = False
        for line in text.splitlines():
            if line.startswith("GUI_THEME="):
                lines.append(f"GUI_THEME={theme_name}")
                found = True
            else:
                lines.append(line)
        if not found:
            lines.append(f"GUI_THEME={theme_name}")
    else:
        # Create minimal .env with just GUI_THEME; onboarding will later
        # rewrite it with API keys when needed.
        lines = [f"GUI_THEME={theme_name}"]

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Update in-memory config so any new build_config() calls see it.
    app_config.GUI_THEME = theme_name
