from __future__ import annotations

from pathlib import Path

from cegsr.utils.io import ensure_dir, write_csv, write_json


def write_run_report(output_dir: str, metrics: dict, examples: list[dict] | None = None) -> None:
    out = ensure_dir(output_dir)
    write_json(out / 'metrics.json', metrics)
    write_csv(out / 'metrics.csv', [metrics])
    md_lines = ['# Run Summary', '', '## Aggregate Metrics']
    for key, value in metrics.items():
        md_lines.append(f'- **{key}**: {value}')
    dataset_rows = [{'name': k.split('::', 1)[1], 'accuracy': v} for k, v in metrics.items() if k.startswith('dataset_accuracy::')]
    if dataset_rows:
        write_csv(out / 'dataset_breakdown.csv', dataset_rows)
        md_lines.extend(['', '## Dataset Breakdown'])
        for row in dataset_rows:
            md_lines.append(f"- {row['name']}: {row['accuracy']}")
    if examples:
        write_json(out / 'error_cases.json', examples)
        write_csv(out / 'error_cases.csv', examples)
        md_lines.extend(['', '## Error Cases'])
        for item in examples:
            md_lines.append(
                f"- sample_id={item['sample_id']} | dataset={item.get('dataset_name','unknown')} | pred={item['pred']} | gold={item['gold']}"
            )
    (Path(out) / 'report.md').write_text('\n'.join(md_lines) + '\n', encoding='utf-8')
