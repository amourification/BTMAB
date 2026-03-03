from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tqdm import tqdm


def main() -> int:
    root = Path(__file__).resolve().parent
    tests_dir = root / "tests"
    print("\n=== Temporal Bot — Test Runner ===\n")

    if not tests_dir.exists():
        print(f"Tests directory not found: {tests_dir}")
        return 1

    test_paths = sorted(tests_dir.glob("test_*.py"))
    if not test_paths:
        print(f"No test_*.py files found in {tests_dir}")
        return 1

    results = []

    with tqdm(total=len(test_paths), desc="Running tests", unit="file") as bar:
        for test_path in test_paths:
            name = test_path.name

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

