#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
from pathlib import Path
from cegsr.config.loader import load_config
from cegsr.workflows import run_end_to_end_method

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output-dir", default="outputs/sirius_lite")
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(run_end_to_end_method(cfg, "sirius_lite", args.output_dir))

if __name__ == "__main__":
    main()
