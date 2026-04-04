#!/usr/bin/env python3
"""
verify_elkies_exact.py - Run the explicit Elkies M23 construction through Sage.

This script does not search a lambda/mu family. It verifies the exact quartic-
field construction encoded in elkies_exact_core.py and records the resulting
mod-prime irreducibility summary.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from elkies_exact_core import build_sage_verification_script

JSON_DIR = Path("testjson")
SAGE_BIN = os.environ.get("SAGE_BIN", "sage")


def run_verification(timeout: int = 300) -> dict:
    JSON_DIR.mkdir(exist_ok=True)
    script = build_sage_verification_script()
    sage_home = (JSON_DIR / ".sage_home").resolve()
    sage_home.mkdir(exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sage", delete=False, encoding="utf-8") as handle:
        handle.write(script)
        script_path = handle.name

    started = time.time()
    try:
        env = os.environ.copy()
        env["HOME"] = str(sage_home)
        env["DOT_SAGE"] = str(sage_home / ".sage")
        result = subprocess.run(
            [SAGE_BIN, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        elapsed = time.time() - started

        payload = {
            "success": result.returncode == 0,
            "elapsed": elapsed,
            "stdout": result.stdout[-6000:],
            "stderr": result.stderr[-2000:],
            "returncode": result.returncode,
            "sage_home": str(sage_home),
        }
        try:
            parsed = json.loads(result.stdout)
        except Exception:
            parsed = None

        if parsed is not None:
            payload["result"] = parsed
        return payload
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def main() -> None:
    payload = run_verification()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = JSON_DIR / f"elkies_exact_results_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Saved Elkies verification results to {output_path}")
    if "result" in payload:
        result = payload["result"]
        print(
            "Irreducible count:",
            f"{result.get('irreducible_count', 0)}/{result.get('tested_count', 0)}",
        )
        print("Consistency score:", result.get("consistency_score", 0.0))
    else:
        print("Sage output was not parseable JSON.")


if __name__ == "__main__":
    main()
