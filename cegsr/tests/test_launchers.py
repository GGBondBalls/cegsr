import json
from pathlib import Path

from cegsr.launchers import generate_experiment_scripts
from cegsr.training.llamafactory_adapter import generate_llamafactory_project


def test_generate_experiment_scripts_writes_vllm_launcher(tmp_path: Path):
    config_dir = tmp_path / "configs" / "profiles"
    config_dir.mkdir(parents=True)
    (tmp_path / "configs" / "base.yaml").write_text(
        "\n".join(
            [
                "project:",
                "  output_dir: outputs/demo",
                "backend:",
                "  kind: mock",
                "task:",
                "  task_type: qa",
                "  dataset_path: outputs/data/demo.jsonl",
                "graph:",
                "  role_order: [planner, solver, verifier, summarizer]",
                "agents: []",
                "credit: {}",
                "repair: {}",
                "experience:",
                "  graph_dir: outputs/demo/graph",
                "training:",
                "  model_name_or_path: /models/Qwen2.5-X.XB-Instruct",
                "  model_size: 7B",
                "evaluation: {}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    config_path = config_dir / "dual.yaml"
    config_path.write_text(
        "\n".join(
            [
                "_base_: ../base.yaml",
                "project:",
                "  output_dir: outputs/dual_4090",
                "serving:",
                "  enabled: true",
                "  kind: vllm",
                "  model_name_or_path: /models/Qwen2.5-X.XB-Instruct",
                "  model_size: 7B",
                "  gpu_ids: [0, 1]",
                "  tensor_parallel_size: 2",
                "  host: 127.0.0.1",
                "  port: 8000",
                "",
            ]
        ),
        encoding="utf-8",
    )

    script_paths = generate_experiment_scripts(config_path, output_dir=str(tmp_path / "outputs" / "dual_4090"))
    serving_script = Path(script_paths["serving"]).read_text(encoding="utf-8")
    assert 'CUDA_VISIBLE_DEVICES="0,1"' in serving_script
    assert "--tensor-parallel-size 2" in serving_script
    assert "/models/Qwen2.5-7B-Instruct" in serving_script
    assert "scripts/run_pipeline.py" in Path(script_paths["pipeline"]).read_text(encoding="utf-8")


def test_generate_llamafactory_project_writes_ddp_script(tmp_path: Path):
    export_dir = tmp_path / "training_data"
    export_dir.mkdir()
    (export_dir / "sft_manifest.json").write_text(json.dumps({"solver": "solver_sft.jsonl"}), encoding="utf-8")
    (export_dir / "solver_sft.jsonl").write_text("[]\n", encoding="utf-8")

    generate_llamafactory_project(
        export_dir=str(export_dir),
        model_name_or_path="/models/Qwen2.5-14B-Instruct",
        output_dir=str(tmp_path / "runs"),
        distributed_config={"gpus": [0, 1], "nproc_per_node": 2, "master_port": 29501},
    )

    ddp_script = (export_dir / "run_llamafactory_ddp.sh").read_text(encoding="utf-8")
    assert 'export CUDA_VISIBLE_DEVICES="0,1"' in ddp_script
    assert 'export FORCE_TORCHRUN=1' in ddp_script
    assert 'export NPROC_PER_NODE="2"' in ddp_script
    assert "llamafactory-cli train" in ddp_script
