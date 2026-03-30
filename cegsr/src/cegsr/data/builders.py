from __future__ import annotations

import random
from collections import defaultdict
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


NORMALIZERS = {
    'gsm8k': _normalize_gsm8k,
    'commonsense_qa': _normalize_commonsenseqa,
    'ai2_arc': _normalize_arc,
    'boolq': _normalize_boolq,
    'pubmed_qa': _normalize_pubmedqa,
}


def _take_split(dataset_name: str, subset: str | None, split: str, limit: int, seed: int) -> list[dict[str, Any]]:
    if dataset_name == 'ai2_arc':
        dataset = _load_dataset(dataset_name, subset or 'ARC-Challenge', split=split)
    elif dataset_name == 'pubmed_qa':
        dataset = _load_dataset('qiaojin/PubMedQA', 'pqa_labeled', split=split)
    else:
        dataset = _load_dataset(dataset_name, split=split)
    rows = list(dataset)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit]


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
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    include_sources = include_sources or ['gsm8k', 'commonsense_qa', 'ai2_arc', 'boolq', 'pubmed_qa']
    split_map = {
        'gsm8k': 'test' if split in {'validation', 'eval', 'test'} else 'train',
        'commonsense_qa': 'validation' if split in {'validation', 'eval', 'test'} else 'train',
        'ai2_arc': 'validation' if split in {'validation', 'eval', 'test'} else 'train',
        'boolq': 'validation' if split in {'validation', 'eval', 'test'} else 'train',
        'pubmed_qa': 'train' if split in {'validation', 'eval', 'test'} else 'train',
    }
    subset_map = {'ai2_arc': 'ARC-Challenge', 'pubmed_qa': 'pqa_labeled'}
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for name in include_sources:
        source_split = split_map[name]
        try:
            source_rows = _take_split(name, subset_map.get(name), source_split, max_per_source, seed)
        except Exception as exc:
            summary_rows.append({'dataset_name': name, 'source_split': source_split, 'num_rows': 0, 'status': f'skipped: {exc.__class__.__name__}'})
            continue
        normalizer = NORMALIZERS[name]
        normalized = [normalizer(row, idx, source_split) for idx, row in enumerate(source_rows)]
        rows.extend(normalized)
        summary_rows.append({'dataset_name': name, 'source_split': source_split, 'num_rows': len(normalized), 'status': 'ok'})

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
