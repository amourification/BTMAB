## Contributing to BTMAB

Thank you for your interest in contributing to **Binance Temporal Market Analysis Bot** (a.k.a. `BTMAB`) maintained by **@amourification**.

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. By contributing, you agree that your contributions are released under the same license.

---

## Getting started

- **Prerequisites**
  - Python **3.11+**
  - `git`

- **Clone and enter the project**

```bash
git clone <your-fork-or-origin-url>
cd "BTMAB"
```

- **Install in editable mode with dev dependencies**

```bash
pip install -e ".[dev]"
```

This installs the main dependencies plus tools like `pytest`, `ruff`, and `black`.

---

## Project layout

- **Source code**: `src/` (packages such as `bot`, `charts`, `consensus`, `data`, `engine`, `gui`, `risk`, `advanced`)
- **Tests**: `tests/` (all `test_*.py` files)
- **CLI entry points**:
  - GUI: `temporal-gui` → `main_gui:main`
  - Telegram bot: `BTMAB` → `bot.telegram_bot:main`

> When adding new modules, put them under the appropriate package in `src/` and add corresponding tests under `tests/`.

---

## Development workflow

1. **Create a feature branch**

```bash
git checkout -b feature/short-description
```

2. **Make changes**
   - Keep functions small and focused.
   - Prefer explicit imports over wildcards.

3. **Run the test suite**

From the `Project` directory:

```bash
python run_all_tests.py
```

This script runs all `tests/test_*.py` files and:
- Skips tests marked `@pytest.mark.slow` by default.
- To include slow tests:

```bash
set INCLUDE_SLOW_TESTS=1  # Windows PowerShell: $env:INCLUDE_SLOW_TESTS = "1"
python run_all_tests.py
```

You can also run `pytest` directly, e.g.:

```bash
pytest tests
```

---

## Code style and quality

- **Formatting**: use `black`

```bash
black .
```

- **Linting**: use `ruff`

```bash
ruff check .
```

Try to keep code free of new linter errors and formatted with `black` before opening a pull request.

---

## Commit and pull requests

- Write **clear, descriptive commit messages**, e.g.:
  - `Refactor risk engine position sizing`
  - `Fix GUI theme persistence on restart`
  - `Add tests for cycle time utilities`

- For pull requests:
  - Describe **what** you changed and **why**.
  - Mention any **backwards-incompatible changes**.
  - Note any tests that are **slow** or require external APIs.

---

## Reporting issues / asking questions

When opening an issue or discussion, please include:

- OS, Python version, and how you installed dependencies.
- Exact command you ran (e.g. `python run_all_tests.py` or `temporal-gui`).
- Full traceback or error messages, plus any relevant logs/screenshots.

---

## License

By contributing to this project, you agree that your contributions are licensed under the **GNU General Public License v3.0 (GPL-3.0)** as described in the `LICENSE` file.

