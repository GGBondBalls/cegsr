from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from cegsr.utils.io import ensure_dir, write_json, write_jsonl


class DatasetPreparationError(RuntimeError):
    pass


def _load_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise DatasetPreparationError(
            'datasets is not installed. Run `pip install datasets` or `pip install -e .[data]`.') from exc
    return load_dataset(*args, **kwargs)


def _choice_lines(choices: list[str]) -> tuple[list[str], str]:
    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    lines = []
    answer = ''
    for idx, choice in enumerate(choices):
        lines.append(f'{labels[idx]}. {choice}')
    return lines, answer


def _choice_labels(num_choices: int) -> list[str]:
    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if num_choices > len(labels):
        raise DatasetPreparationError(f'Unsupported number of choices: {num_choices}')
    return [labels[idx] for idx in range(num_choices)]


def _format_choice_answer(answer_label: str | None, answer_text: str | None, fallback: str = '') -> str:
    if answer_label and answer_text:
        return f'{answer_label}. {answer_text}'
    if answer_label:
        return answer_label
    if answer_text:
        return answer_text
    return fallback


def _resolve_choice_answer(answer_value: Any, choice_texts: list[str]) -> tuple[str | None, str | None]:
    labels = _choice_labels(len(choice_texts))
    normalized_texts = [str(choice).strip() for choice in choice_texts]

    if isinstance(answer_value, bool):
        answer_value = int(answer_value)

    if isinstance(answer_value, int):
        if 0 <= answer_value < len(labels):
            return labels[answer_value], normalized_texts[answer_value]
        return None, None

    raw = str(answer_value).strip()
    if not raw:
        return None, None

    upper = raw.upper()
    if upper in labels:
        idx = labels.index(upper)
        return upper, normalized_texts[idx]

    if raw.isdigit():
        idx = int(raw)
        if 0 <= idx < len(labels):
            return labels[idx], normalized_texts[idx]

    for label, text in zip(labels, normalized_texts):
        if raw == text:
            return label, text
    return None, None


def _normalize_gsm8k(record: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
    answer = str(record.get('answer', ''))
    final_answer = answer.split('####')[-1].strip() if '####' in answer else answer.strip()
    return {
        'sample_id': f'gsm8k_{split}_{idx}',
        'question': str(record.get('question', '')).strip(),
        'answer': final_answer,
        'context': '',
        'choices': [],
        'task_type': 'qa',
        'metadata': {
            'dataset_name': 'gsm8k',
            'source_split': split,
            'category': 'math_word_problem',
            'rationale': answer,
        },
    }


def _normalize_commonsenseqa(record: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
    question = str(record.get('question', '')).strip()
    choice_stems = [c.strip() for c in record.get('choices', {}).get('text', [])]
    choice_labels = [c.strip() for c in record.get('choices', {}).get('label', [])]
    labeled_choices = [f'{label}. {text}' for label, text in zip(choice_labels, choice_stems)]
    answer_key = str(record.get('answerKey', '')).strip()
    answer_text = ''
    for label, text in zip(choice_labels, choice_stems):
        if label == answer_key:
            answer_text = f'{label}. {text}'
            break
    return {
        'sample_id': f'commonsenseqa_{split}_{idx}',
        'question': question,
        'answer': answer_text or answer_key,
        'context': '',
        'choices': labeled_choices,
        'task_type': 'mmlu_style',
        'metadata': {
            'dataset_name': 'commonsense_qa',
            'source_split': split,
            'category': 'commonsense',
            'answer_label': answer_key,
        },
    }


def _normalize_arc(record: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
    question = str(record.get('question', '')).strip()
    choices = record.get('choices', {})
    labels = [str(x).strip() for x in choices.get('label', [])]
    texts = [str(x).strip() for x in choices.get('text', [])]
    labeled_choices = [f'{label}. {text}' for label, text in zip(labels, texts)]
    answer_key = str(record.get('answerKey', '')).strip()
    answer_text = ''
    for label, text in zip(labels, texts):
        if label == answer_key:
            answer_text = f'{label}. {text}'
            break
    return {
        'sample_id': f'arc_{split}_{idx}',
        'question': question,
        'answer': answer_text or answer_key,
        'context': '',
        'choices': labeled_choices,
        'task_type': 'mmlu_style',
        'metadata': {
            'dataset_name': 'ai2_arc',
            'subset': 'ARC-Challenge',
            'source_split': split,
            'category': 'science_mcq',
            'answer_label': answer_key,
        },
    }


def _normalize_mmlu_subject(
    record: dict[str, Any],
    idx: int,
    split: str,
    dataset_name: str,
    subject: str,
) -> dict[str, Any]:
    question = str(record.get('question', '')).strip()
    choices = [str(choice).strip() for choice in record.get('choices', [])]
    labels = _choice_labels(len(choices))
    labeled_choices = [f'{label}. {choice}' for label, choice in zip(labels, choices)]
    answer_label, answer_text = _resolve_choice_answer(record.get('answer'), choices)
    answer = _format_choice_answer(answer_label, answer_text, fallback=str(record.get('answer', '')).strip())
    return {
        'sample_id': f'{dataset_name}_{split}_{idx}',
        'question': question,
        'answer': answer,
        'context': '',
        'choices': labeled_choices,
        'task_type': 'mmlu_style',
        'metadata': {
            'dataset_name': dataset_name,
            'source_split': split,
            'category': subject,
            'mmlu_subject': subject,
            'answer_label': answer_label or '',
        },
    }


def _normalize_boolq(record: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
    question = str(record.get('question', '')).strip()
    context = str(record.get('passage', '')).strip()
    answer = 'yes' if bool(record.get('answer')) else 'no'
    return {
        'sample_id': f'boolq_{split}_{idx}',
        'question': question,
        'answer': answer,
        'context': context,
        'choices': ['A. yes', 'B. no'],
        'task_type': 'qa',
        'metadata': {
            'dataset_name': 'boolq',
            'source_split': split,
            'category': 'reading_comprehension_yesno',
        },
    }


def _normalize_pubmedqa(record: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
    question = str(record.get('question', '')).strip()
    context = ' '.join(str(x).strip() for x in record.get('context', {}).get('contexts', []) if str(x).strip())
    long_answer = str(record.get('long_answer', '')).strip()
    answer = str(record.get('final_decision', '')).strip().lower()
    return {
        'sample_id': f'pubmedqa_{split}_{idx}',
        'question': question,
        'answer': answer,
        'context': context,
        'choices': ['A. yes', 'B. no', 'C. maybe'],
        'task_type': 'pubmedqa_style',
        'metadata': {
            'dataset_name': 'pubmed_qa',
            'source_split': split,
            'category': 'biomedical_qa',
            'long_answer': long_answer,
        },
    }


def _normalize_college_physics(record: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
    return _normalize_mmlu_subject(
        record,
        idx=idx,
        split=split,
        dataset_name='college_physics',
        subject='college_physics',
    )


def _normalize_college_chemistry(record: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
    return _normalize_mmlu_subject(
        record,
        idx=idx,
        split=split,
        dataset_name='college_chemistry',
        subject='college_chemistry',
    )


NORMALIZERS = {
    'gsm8k': _normalize_gsm8k,
    'commonsense_qa': _normalize_commonsenseqa,
    'ai2_arc': _normalize_arc,
    'boolq': _normalize_boolq,
    'pubmed_qa': _normalize_pubmedqa,
    'college_physics': _normalize_college_physics,
    'college_chemistry': _normalize_college_chemistry,
}


DATASET_SPECS: dict[str, dict[str, Any]] = {
    'gsm8k': {
        'sources': [
            {'path': 'openai/gsm8k', 'config': 'main'},
            {'path': 'gsm8k', 'config': 'main'},
        ],
        'eval_splits': ['test', 'validation'],
        'train_splits': ['train'],
    },
    'commonsense_qa': {
        'sources': [
            {'path': 'tau/commonsense_qa'},
            {'path': 'commonsense_qa'},
        ],
        'eval_splits': ['validation'],
        'train_splits': ['train'],
    },
    'ai2_arc': {
        'sources': [
            {'path': 'allenai/ai2_arc', 'config': 'ARC-Challenge'},
            {'path': 'ai2_arc', 'config': 'ARC-Challenge'},
        ],
        'eval_splits': ['validation'],
        'train_splits': ['train'],
    },
    'boolq': {
        'sources': [
            {'path': 'google/boolq'},
            {'path': 'super_glue', 'config': 'boolq'},
            {'path': 'boolq'},
        ],
        'eval_splits': ['validation'],
        'train_splits': ['train'],
    },
    'pubmed_qa': {
        'sources': [
            {'path': 'qiaojin/PubMedQA', 'config': 'pqa_labeled'},
        ],
        'eval_splits': ['train'],
        'train_splits': ['train'],
    },
    'college_physics': {
        'sources': [
            {'path': 'cais/mmlu', 'config': 'college_physics'},
            {'path': 'lukaemon/mmlu', 'config': 'college_physics'},
        ],
        'eval_splits': ['test', 'validation', 'dev'],
        'train_splits': ['auxiliary_train', 'dev', 'validation'],
    },
    'college_chemistry': {
        'sources': [
            {'path': 'cais/mmlu', 'config': 'college_chemistry'},
            {'path': 'lukaemon/mmlu', 'config': 'college_chemistry'},
        ],
        'eval_splits': ['test', 'validation', 'dev'],
        'train_splits': ['auxiliary_train', 'dev', 'validation'],
    },
}


def _split_candidates_for(dataset_name: str, split: str) -> list[str]:
    spec = DATASET_SPECS.get(dataset_name)
    if spec is None:
        raise DatasetPreparationError(f'Unsupported dataset source: {dataset_name}')
    bucket = 'train_splits' if split.lower() == 'train' else 'eval_splits'
    return list(spec[bucket])


def _take_split(dataset_name: str, split: str, limit: int, seed: int) -> tuple[list[dict[str, Any]], str, str]:
    spec = DATASET_SPECS.get(dataset_name)
    if spec is None:
        raise DatasetPreparationError(f'Unsupported dataset source: {dataset_name}')

    errors: list[str] = []
    for source in spec['sources']:
        source_path = source['path']
        source_config = source.get('config')
        for split_name in _split_candidates_for(dataset_name, split):
            try:
                if source_config:
                    dataset = _load_dataset(source_path, source_config, split=split_name)
                else:
                    dataset = _load_dataset(source_path, split=split_name)
                rows = list(dataset)
                rng = random.Random(seed)
                rng.shuffle(rows)
                return rows[:limit], split_name, source_path
            except Exception as exc:
                source_ref = f'{source_path}/{source_config}' if source_config else source_path
                errors.append(f'{source_ref}:{split_name}:{exc.__class__.__name__}')
                continue
    raise DatasetPreparationError('; '.join(errors))


def build_reasoning_mix(
    output_path: str | Path,
    split: str = 'validation',
    max_per_source: int = 100,
    seed: int = 42,
    include_sources: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build a unified reasoning benchmark from public datasets.

    Default sources:
    - gsm8k
    - commonsense_qa
    - ai2_arc (ARC-Challenge)
    - boolq
    - pubmed_qa (optional; skipped if unavailable)
    - college_physics
    - college_chemistry
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    include_sources = include_sources or [
        'gsm8k',
        'commonsense_qa',
        'ai2_arc',
        'boolq',
        'pubmed_qa',
        'college_physics',
        'college_chemistry',
    ]
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for name in include_sources:
        try:
            source_rows, source_split, source_ref = _take_split(name, split, max_per_source, seed)
        except Exception as exc:
            requested_splits = ','.join(_split_candidates_for(name, split))
            summary_rows.append(
                {
                    'dataset_name': name,
                    'source_split': requested_splits,
                    'num_rows': 0,
                    'status': f'skipped: {exc.__class__.__name__}',
                }
            )
            continue
        normalizer = NORMALIZERS[name]
        normalized = [normalizer(row, idx, source_split) for idx, row in enumerate(source_rows)]
        rows.extend(normalized)
        summary_rows.append(
            {
                'dataset_name': name,
                'source_split': source_split,
                'num_rows': len(normalized),
                'source_ref': source_ref,
                'status': 'ok',
            }
        )

    write_jsonl(output_path, rows)
    summary = {
        'output_path': str(output_path),
        'num_rows': len(rows),
        'split': split,
        'max_per_source': max_per_source,
        'seed': seed,
        'sources': summary_rows,
    }
    write_json(output_path.with_suffix('.meta.json'), summary)
    return summary


def prepare_dataset(recipe: str, output_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    recipe = recipe.lower()
    if recipe == 'reasoning_mix':
        return build_reasoning_mix(output_path=output_path, **kwargs)
    raise ValueError(f'Unsupported recipe: {recipe}')
