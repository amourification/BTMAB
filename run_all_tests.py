from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tqdm import tqdm


TEST_FILES = [
    "test_stops_utils.py",
    "test_elliott_fib.py",
    "test_cycle_time_utils.py",
    "test_trade_suggestions.py",
    "test_integration.py",
    "test_backtest_binance_range.py",
]


def main() -> int:
    root = Path(__file__).resolve().parent
    print("\n=== Temporal Bot — Test Runner ===\n")

    results = []

    with tqdm(total=len(TEST_FILES), desc="Running tests", unit="file") as bar:
        for name in TEST_FILES:
            test_path = root / name
            if not test_path.exists():
                results.append((name, False, "", f"File not found: {test_path}"))
                bar.update(1)
                continue

            bar.set_description(f"{name}")
            # By default, skip tests marked as \"slow\" (e.g. live backtests).
            # Set INCLUDE_SLOW_TESTS=1 in the environment to include them.
            import os
            include_slow = os.getenv("INCLUDE_SLOW_TESTS") == "1"
            markers = [] if include_slow else ["-m", "not slow"]

            cmd = [sys.executable, "-m", "pytest", str(test_path), "-q", *markers]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            ok = proc.returncode == 0
            results.append((name, ok, proc.stdout, proc.stderr))
            bar.update(1)

    print("\n=== Test Summary ===")
    all_ok = True
    for name, ok, out, err in results:
        status = "OK" if ok else "FAIL"
        print(f"- {name}: {status}")
        if not ok:
            all_ok = False
            if err.strip():
                print(f"  stderr:\n{err.strip()}\n")
            elif out.strip():
                print(f"  output:\n{out.strip()}\n")

    if all_ok:
        print("✅ All tests passed.")
        return 0

    print("❌ Some tests failed. See details above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

