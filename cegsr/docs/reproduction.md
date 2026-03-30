# Reproduction

## Smoke test

```bash
python scripts/run_ablation.py --config configs/base_demo.yaml --output-dir outputs/ablations_demo
```

## Real local experiment

```bash
python scripts/prepare_data.py --config configs/datasets/reasoning_mix_eval.yaml
python scripts/run_ablation.py --config configs/base.yaml --output-dir outputs/ablations_real
```

## Local Qwen path convention

The project defaults to a cache-root style path:

```text
E:/home/fyk/models/Qwen/Qwen2.5-X.XB-Instruct
```

Change `model_size` in the config instead of hardcoding the snapshot path by hand.
