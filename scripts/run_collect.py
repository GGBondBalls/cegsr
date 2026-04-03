#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
from cegsr.workflows import collect_episodes
from cegsr.utils.logging import setup_logging

def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--use-retrieval", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--verbose-turns", action="store_true")
    args = parser.parse_args()
    print(
        collect_episodes(
            args.config,
            output_path=args.output,
            use_retrieval=args.use_retrieval,
            max_samples=args.max_samples,
            verbose_turns=args.verbose_turns,
        )
    )

if __name__ == "__main__":
    main()
