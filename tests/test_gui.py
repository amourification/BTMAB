from __future__ import annotations

"""
test_gui.py — Simple Tkinter UI for running the test suite.

Features:
- Checkboxes to select which test modules to run.
- Option to include or skip slow tests (e.g. live Binance backtests).
- Progress bar across selected tests.
- Terminal-style output window showing pytest output per test file.
- Short summary explaining results.
"""

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import List

try:
    import tkinter as tk
    from tkinter import ttk
except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
    raise SystemExit(
        "Tkinter is not available in this Python installation.\n"
        "Install the system tk packages (e.g. `sudo apt install python3-tk`) and retry."
    ) from exc


ROOT = Path(__file__).resolve().parent


def _default_test_files() -> List[str]:
    """
    Try to reuse TEST_FILES from run_all_tests.py so the GUI stays in sync.
    Falls back to a hard-coded list if that import fails (e.g. missing tqdm).
    """
    try:
        from run_all_tests import TEST_FILES  # type: ignore[attr-defined]

        return list(TEST_FILES)
    except Exception:
        return [
            "test_stops_utils.py",
            "test_elliott_fib.py",
            "test_cycle_time_utils.py",
            "test_trade_suggestions.py",
            "test_integration.py",
            "test_backtest_binance_range.py",
        ]


class TestRunnerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Temporal Bot — Test Runner")
        self.geometry("900x600")

        self._tests = _default_test_files()
        self._test_vars: List[tk.BooleanVar] = []
        self._include_slow = tk.BooleanVar(value=False)
        self._running = False

        self._build_layout()

    # --------------------------------------------------------------------- UI
    def _build_layout(self) -> None:
        main = ttk.PanedWindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=8, pady=8)

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=1)
        main.add(right, weight=3)

        # Left: test selection + options
        ttk.Label(left, text="Select tests to run:", font=("Segoe UI", 10, "bold")).pack(
            anchor="w"
        )

        tests_frame = ttk.Frame(left)
        tests_frame.pack(fill="both", expand=False, pady=(4, 8))

        for name in self._tests:
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(tests_frame, text=name, variable=var)
            cb.pack(anchor="w")
            self._test_vars.append(var)

        opts = ttk.Frame(left)
        opts.pack(fill="x", pady=(4, 4))

        ttk.Checkbutton(
            opts,
            text="Include slow tests (live/backtests)",
            variable=self._include_slow,
        ).pack(anchor="w")

        self._run_btn = ttk.Button(left, text="Run Selected Tests", command=self._on_run)
        self._run_btn.pack(fill="x", pady=(8, 4))

        # Progress bar and status
        self._progress = ttk.Progressbar(left, mode="determinate", maximum=100)
        self._progress.pack(fill="x", pady=(4, 2))

        self._status_var = tk.StringVar(value="Idle")
        ttk.Label(left, textvariable=self._status_var).pack(anchor="w")

        # Short legend for result meaning
        legend = (
            "Results meaning:\n"
            "- returncode 0: all tests in the module passed.\n"
            "- non-zero returncode: some tests failed (see output below).\n"
            "- Slow tests usually hit live APIs and can be much slower."
        )
        ttk.Label(left, text=legend, wraplength=260, justify="left").pack(
            anchor="w", pady=(8, 0)
        )

        # Right: terminal-style output
        ttk.Label(right, text="Test output:", font=("Segoe UI", 10, "bold")).pack(
            anchor="w"
        )
        txt_frame = ttk.Frame(right)
        txt_frame.pack(fill="both", expand=True)

        self._text = tk.Text(
            txt_frame,
            wrap="word",
            state="disabled",
            font=("Consolas", 9),
        )
        yscroll = ttk.Scrollbar(txt_frame, orient="vertical", command=self._text.yview)
        self._text.configure(yscrollcommand=yscroll.set)
        self._text.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

    # ----------------------------------------------------------------- Helpers
    def _append_output(self, text: str) -> None:
        self._text.configure(state="normal")
        self._text.insert("end", text + "\n")
        self._text.see("end")
        self._text.configure(state="disabled")

    def _on_run(self) -> None:
        if self._running:
            return
        selected = [
            name for name, var in zip(self._tests, self._test_vars) if var.get()
        ]
        if not selected:
            self._status_var.set("No tests selected.")
            return
        self._running = True
        self._run_btn.configure(state="disabled")
        self._progress["value"] = 0
        self._append_output("=" * 60)
        self._append_output("Starting test run...")

        thread = threading.Thread(
            target=self._run_tests_thread, args=(selected, self._include_slow.get())
        )
        thread.daemon = True
        thread.start()

    def _run_tests_thread(self, tests: List[str], include_slow: bool) -> None:
        total = len(tests)
        passed = 0

        # Prefer the project virtualenv's Python if it exists, so tests
        # always see the same dependencies as the main app.
        venv_candidates = [
            ROOT / ".venv" / "Scripts" / "python.exe",   # Windows
            ROOT / ".venv" / "bin" / "python3",          # Unix
            ROOT / ".venv" / "bin" / "python",
        ]
        runner = None
        for cand in venv_candidates:
            if cand.exists():
                runner = str(cand)
                break
        if runner is None:
            runner = sys.executable

        for idx, name in enumerate(tests, start=1):
            self._status_var.set(f"Running {name} ({idx}/{total})...")
            markers = [] if include_slow else ["-m", "not slow"]
            cmd = [runner, "-m", "pytest", name, "-q", *markers]

            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                )
                ok = proc.returncode == 0
                if ok:
                    passed += 1
                header = f"[{name}] returncode={proc.returncode}"
                self._append_output(header)
                if proc.stdout.strip():
                    self._append_output(proc.stdout.strip())
                if proc.stderr.strip():
                    self._append_output("stderr:\n" + proc.stderr.strip())
            except Exception as exc:
                self._append_output(f"[{name}] ERROR: {exc}")

            progress = int(idx / total * 100)
            self._progress["value"] = progress

        summary = f"Finished: {passed}/{total} test modules passed."
        self._status_var.set(summary)
        self._append_output(summary)
        self._running = False
        self._run_btn.configure(state="normal")


def main() -> None:
    app = TestRunnerApp()
    app.mainloop()


if __name__ == "__main__":
    main()

