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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="outputs/training_data")
    args = parser.parse_args()
    script = Path(args.config_dir) / "run_llamafactory.sh"
    print(f"Generated helper script: {script}")

if __name__ == "__main__":
    main()
