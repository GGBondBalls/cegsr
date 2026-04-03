#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
from cegsr.config.loader import load_config
from cegsr.data.builders import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description='Build unified CEG-SR datasets from public benchmarks.')
    parser.add_argument('--config', default='configs/datasets/reasoning_mix_eval.yaml')
    parser.add_argument('--recipe', default=None)
    parser.add_argument('--output-path', default=None)
    parser.add_argument('--max-per-source', type=int, default=None)
    parser.add_argument('--split', default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    recipe = args.recipe or cfg.get('recipe', 'reasoning_mix')
    output_path = args.output_path or cfg['output_path']
    summary = prepare_dataset(
        recipe=recipe,
        output_path=output_path,
        split=args.split or cfg.get('split', 'validation'),
        max_per_source=args.max_per_source or int(cfg.get('max_per_source', 100)),
        seed=args.seed or int(cfg.get('seed', 42)),
        include_sources=cfg.get('include_sources'),
    )
    print(summary)


if __name__ == '__main__':
    main()
