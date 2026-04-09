# Iterative Train Analysis Artifacts

This directory stores the commit-friendly analysis subset of the iterative run launched with:

```bash
python scripts/run_iterative.py \
  --config configs/profiles/single_vgpu32.yaml \
  --max-iterations 3 \
  --training-mode qlora \
  --early-stop 2
```

Included here:

- top-level run summaries (`iterative_summary.json`, `iteration_curve.csv`, `best_adapters.json`)
- per-iteration raw / annotated / repaired train+eval episode files
- per-iteration eval reports
- per-iteration exported training data (`*_sft.jsonl`, `preference_pairs.jsonl`, `filter_stats.json`, YAML configs)
- per-role training logs and result summaries

Excluded on purpose:

- adapter weights
- tokenizer copies
- checkpoints
- optimizer / scheduler / RNG state

The full original run is still kept on the data disk:

- `/root/autodl-tmp/cegsr_outputs/iterative_train`

This repo copy exists so the analysis artifacts live under the canonical project output path and can be committed directly.
