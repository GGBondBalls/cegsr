from pathlib import Path

from cegsr.utils.modeling import render_model_path_template, resolve_local_model_path


def test_render_model_path_template():
    rendered = render_model_path_template("/models/Qwen2.5-X.XB-Instruct", model_size_hint="14B")
    assert rendered == "/models/Qwen2.5-14B-Instruct"


def test_resolve_local_model_path_snapshot(tmp_path: Path):
    repo = tmp_path / 'models--Qwen--Qwen2.5-7B-Instruct'
    snap = repo / 'snapshots' / '123abc'
    snap.mkdir(parents=True)
    (snap / 'config.json').write_text('{}', encoding='utf-8')
    (snap / 'tokenizer_config.json').write_text('{}', encoding='utf-8')
    resolved = resolve_local_model_path(str(repo))
    assert Path(resolved) == snap
