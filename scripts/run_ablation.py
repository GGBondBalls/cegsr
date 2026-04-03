#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
from cegsr.workflows import run_ablation_suite

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Only run specified methods, e.g. --methods sirius_lite ours_wo_graph")
    args = parser.parse_args()
    print(run_ablation_suite(args.config, output_dir=args.output_dir, methods=args.methods))

if __name__ == "__main__":
    main()
