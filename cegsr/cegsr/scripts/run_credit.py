#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
from cegsr.workflows import annotate_episodes

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--trajectory-level-only", action="store_true")
    args = parser.parse_args()
    print(annotate_episodes(args.episodes, args.config, output_path=args.output, trajectory_level_only=args.trajectory_level_only))

if __name__ == "__main__":
    main()
