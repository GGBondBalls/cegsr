#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
from cegsr.workflows import build_experience_graph

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--graph-dir", default=None)
    args = parser.parse_args()
    print(build_experience_graph(args.episodes, args.config, graph_dir=args.graph_dir))

if __name__ == "__main__":
    main()
