#!/usr/bin/env python
"""
Run the iterative self-improvement loop.

Usage:
    python scripts/run_iterative.py --config configs/profiles/dual_4090_vllm_paper.yaml
    python scripts/run_iterative.py --config configs/profiles/dual_4090_vllm_paper.yaml --max-iterations 5 --mode qlora --dpo
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

from cegsr.workflows import run_iterative


def main() -> None:
    parser = argparse.ArgumentParser(description="CEG-SR iterative self-improvement loop")
    parser.add_argument("--config", default="configs/profiles/dual_4090_vllm_paper.yaml",
                        help="Path to experiment config")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Maximum number of self-improvement iterations")
    parser.add_argument("--mode", choices=["lora", "qlora"], default=None,
                        help="Training mode (default: from config or qlora)")
    parser.add_argument("--dpo", action="store_true",
                        help="Run DPO stage after SFT")
    parser.add_argument("--credit-threshold", type=float, default=0.65,
                        help="High credit threshold for training data selection")
    parser.add_argument("--use-eval-split", action="store_true",
                        help="Use eval split for training (when no separate train split)")
    args = parser.parse_args()

    summary = run_iterative(
        config_or_path=args.config,
        output_dir=args.output_dir,
        max_iterations=args.max_iterations,
        training_mode=args.mode,
        run_dpo=args.dpo,
        high_credit_threshold=args.credit_threshold,
        use_train_split=not args.use_eval_split,
    )

    print(f"\n{'='*60}")
    print(f"Iterative training complete: {summary['total_iterations']} iterations")
    print(f"Final eval accuracy: {summary['final_accuracy']:.4f}")
    for it in summary['iterations']:
        acc = it['eval_metrics'].get('accuracy', 0)
        print(f"  Iteration {it['iteration']}: accuracy={acc:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
