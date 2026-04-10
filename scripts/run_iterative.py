#!/usr/bin/env python
"""CLI entry point for the iterative self-improvement loop."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

from cegsr.workflows import run_iterative_loop


def main() -> None:
    parser = argparse.ArgumentParser(description="CEG-SR Iterative Self-Improvement Loop")
    parser.add_argument("--config", default="configs/base.yaml", help="Config file path")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--max-iterations", type=int, default=None, help="Max training iterations")
    parser.add_argument("--training-mode", choices=["lora", "qlora"], default=None, help="Training mode")
    parser.add_argument("--dpo", action="store_true", default=None, help="Enable DPO after SFT")
    parser.add_argument("--early-stop", type=int, default=2, help="Early stop patience")
    parser.add_argument(
        "--credit-mode",
        choices=["fine_grained", "trajectory"],
        default="fine_grained",
        help="Credit mode: fine_grained (CEG-SR) or trajectory (SiriuS-SFT baseline)",
    )
    parser.add_argument(
        "--no-cross-iter-mix",
        action="store_true",
        default=False,
        help="Disable cross-iteration data accumulation",
    )
    args = parser.parse_args()

    summary = run_iterative_loop(
        config_or_path=args.config,
        output_dir=args.output_dir,
        max_iterations=args.max_iterations,
        training_mode=args.training_mode,
        dpo_enabled=args.dpo,
        early_stop_patience=args.early_stop,
        credit_mode=args.credit_mode,
        cross_iteration_mix=not args.no_cross_iter_mix,
    )

    baseline = summary.get('baseline', {})
    print(f"\nIterative loop complete ({summary.get('credit_mode', 'fine_grained')}):")
    print(f"  Baseline: raw={baseline.get('raw_accuracy', 0):.4f}")
    print(f"  Total iterations: {summary['total_iterations']}")
    print(f"  Best accuracy: {summary['best_accuracy']:.4f}")
    for r in summary['iteration_results']:
        print(f"  Iter {r['iteration']}: raw={r['raw_accuracy']:.4f}  "
              f"repaired={r['accuracy']:.4f}  "
              f"sft={r['export_stats'].get('sft_selected', 0)}  "
              f"sel_rate={r['export_stats'].get('selection_rate', 0)*100:.1f}%  "
              f"pref_pairs={r['export_stats'].get('preference_pairs', 0)}")


if __name__ == "__main__":
    main()
