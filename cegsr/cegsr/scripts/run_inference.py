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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", default="outputs/inference_episodes.jsonl")
    args = parser.parse_args()
    print(collect_episodes(args.config, output_path=args.output, use_retrieval=False))

if __name__ == "__main__":
    main()
