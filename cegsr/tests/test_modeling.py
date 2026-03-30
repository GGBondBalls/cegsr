from pathlib import Path

from cegsr.utils.modeling import resolve_local_model_path


def test_resolve_local_model_path_snapshot(tmp_path: Path):
    repo = tmp_path / 'models--Qwen--Qwen2.5-7B-Instruct'
    snap = repo / 'snapshots' / '123abc'
    snap.mkdir(parents=True)
    (snap / 'config.json').write_text('{}', encoding='utf-8')
    (snap / 'tokenizer_config.json').write_text('{}', encoding='utf-8')
    resolved = resolve_local_model_path(str(repo))
    assert Path(resolved) == snap
