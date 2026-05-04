"""Minimal benchmark entry point for Ferrum release packaging."""

from __future__ import annotations

import time


def main() -> None:
    start = time.perf_counter()
    time.sleep(0.0)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"Benchmark placeholder completed in {elapsed_ms:.3f} ms")


if __name__ == "__main__":
    main()
