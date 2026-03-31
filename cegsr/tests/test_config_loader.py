from pathlib import Path

from cegsr.config.loader import load_config
from cegsr.utils.modeling import render_model_path_template


def test_load_config_supports_base_inheritance(tmp_path: Path):
    base_path = tmp_path / "base.yaml"
    child_path = tmp_path / "profiles" / "child.yaml"
    base_path.write_text(
        "\n".join(
            [
                "project:",
                "  output_dir: outputs/demo",
                "backend:",
                "  kind: hf_local",
                "  model_name_or_path: /models/Qwen2.5-X.XB-Instruct",
                "  model_size: 7B",
                "",
            ]
        ),
        encoding="utf-8",
    )
    child_path.parent.mkdir(parents=True)
    child_path.write_text(
        "\n".join(
            [
                "_base_: ../base.yaml",
                "backend:",
                "  kind: vllm",
                "  model_size: 14B",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(child_path)
    assert config["project"]["output_dir"] == "outputs/demo"
    assert config["backend"]["kind"] == "vllm"
    assert config["backend"]["model_size"] == "14B"
    assert render_model_path_template(config["backend"]["model_name_or_path"], config["backend"]["model_size"]) == "/models/Qwen2.5-14B-Instruct"
