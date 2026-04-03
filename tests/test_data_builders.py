from pathlib import Path

from cegsr.data import builders
from cegsr.utils.io import read_jsonl


def test_build_reasoning_mix_loads_gsm8k_main_with_validation_fallback(monkeypatch, tmp_path: Path):
    calls: list[tuple[str, str | None, str]] = []

    def fake_load_dataset(path, config=None, split=None):
        calls.append((path, config, split))
        if (path, config, split) == ('openai/gsm8k', 'main', 'validation'):
            return [{'question': '1+1=?', 'answer': 'Reasoning #### 2'}]
        raise ValueError('missing split')

    monkeypatch.setattr(builders, '_load_dataset', fake_load_dataset)

    summary = builders.build_reasoning_mix(
        output_path=tmp_path / 'mix.jsonl',
        split='validation',
        max_per_source=1,
        seed=0,
        include_sources=['gsm8k'],
    )

    rows = read_jsonl(tmp_path / 'mix.jsonl')
    assert summary['num_rows'] == 1
    assert rows[0]['answer'] == '2'
    assert ('openai/gsm8k', 'main', 'test') in calls
    assert ('openai/gsm8k', 'main', 'validation') in calls


def test_build_reasoning_mix_falls_back_to_super_glue_boolq(monkeypatch, tmp_path: Path):
    class HfHubHTTPError(RuntimeError):
        pass

    calls: list[tuple[str, str | None, str]] = []

    def fake_load_dataset(path, config=None, split=None):
        calls.append((path, config, split))
        if path == 'google/boolq':
            raise HfHubHTTPError('blocked')
        if (path, config, split) == ('super_glue', 'boolq', 'validation'):
            return [{'question': 'Is water wet?', 'passage': 'Water is wet.', 'answer': True}]
        raise ValueError('unexpected path')

    monkeypatch.setattr(builders, '_load_dataset', fake_load_dataset)

    summary = builders.build_reasoning_mix(
        output_path=tmp_path / 'mix.jsonl',
        split='validation',
        max_per_source=1,
        seed=0,
        include_sources=['boolq'],
    )

    rows = read_jsonl(tmp_path / 'mix.jsonl')
    assert summary['num_rows'] == 1
    assert rows[0]['answer'] == 'yes'
    assert ('google/boolq', None, 'validation') in calls
    assert ('super_glue', 'boolq', 'validation') in calls


def test_build_reasoning_mix_supports_college_subjects(monkeypatch, tmp_path: Path):
    def fake_load_dataset(path, config=None, split=None):
        if (path, config, split) == ('cais/mmlu', 'college_physics', 'test'):
            return [{'question': 'A force acts on a mass. What changes?', 'choices': ['velocity', 'color', 'mass', 'charge'], 'answer': 0}]
        if (path, config, split) == ('cais/mmlu', 'college_chemistry', 'test'):
            return [{'question': 'What is the pH of pure water?', 'choices': ['1', '7', '10', '14'], 'answer': 'B'}]
        raise ValueError('unexpected path')

    monkeypatch.setattr(builders, '_load_dataset', fake_load_dataset)

    summary = builders.build_reasoning_mix(
        output_path=tmp_path / 'paper_mix.jsonl',
        split='validation',
        max_per_source=1,
        seed=0,
        include_sources=['college_physics', 'college_chemistry'],
    )

    rows = read_jsonl(tmp_path / 'paper_mix.jsonl')
    assert summary['num_rows'] == 2
    assert rows[0]['task_type'] == 'mmlu_style'
    assert rows[0]['answer'].startswith('A. ')
    assert rows[0]['metadata']['dataset_name'] == 'college_physics'
    assert rows[1]['task_type'] == 'mmlu_style'
    assert rows[1]['answer'] == 'B. 7'
    assert rows[1]['metadata']['dataset_name'] == 'college_chemistry'
