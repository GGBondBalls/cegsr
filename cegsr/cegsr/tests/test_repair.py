from cegsr.workflows import build_system, load_samples
from cegsr.credit.outcome_credit import OutcomeCreditSignal
from cegsr.credit.verifier_credit import VerifierCreditSignal
from cegsr.credit.dependency_credit import DependencyCreditSignal
from cegsr.credit.fusion import fuse_credit_records
from cegsr.repair.selective_repair import SelectiveRepairEngine
from cegsr.trajectories.schema import ExperienceNode
from cegsr.trajectories.segmentation import segment_episode


def test_selective_repair_runs():
    cfg = {
        "project": {"output_dir": "outputs/test"},
        "task": {"task_type": "qa", "dataset_path": "examples/sample_dataset.jsonl"},
        "backend": {"kind": "mock"},
        "graph": {"role_order": ["planner", "solver", "verifier", "summarizer"]},
        "agents": [{"role": "planner"}, {"role": "solver"}, {"role": "verifier"}, {"role": "summarizer"}, {"role":"single_agent"}],
        "credit": {"weights": {"outcome": 0.4, "verifier": 0.4, "dependency": 0.2}},
        "repair": {"turn_threshold": 0.95, "subtrajectory_threshold": 0.95},
        "experience": {"graph_dir": "outputs/test/graph", "min_credit": 0.6},
        "training": {"model_name_or_path": "local-model"},
        "evaluation": {},
    }
    system = build_system(cfg, use_graph=False)
    sample = load_samples("examples/sample_dataset.jsonl")[0]
    episode = system["runtime"].run_sample(sample)
    segment_episode(episode)
    groups = [OutcomeCreditSignal().compute(episode), VerifierCreditSignal().compute(episode), DependencyCreditSignal().compute(episode)]
    fuse_credit_records(episode, groups)
    repairer = SelectiveRepairEngine(system["runtime"], turn_threshold=0.99, subtrajectory_threshold=0.99)
    repaired = repairer.repair(episode)
    assert repaired.final_prediction


def test_runtime_only_retrieves_for_enabled_roles():
    cfg = {
        "project": {"output_dir": "outputs/test"},
        "task": {"task_type": "qa", "dataset_path": "examples/sample_dataset.jsonl"},
        "backend": {"kind": "mock"},
        "graph": {"role_order": ["planner", "solver", "verifier", "summarizer"]},
        "agents": [{"role": "planner"}, {"role": "solver"}, {"role": "verifier"}, {"role": "summarizer"}, {"role": "single_agent"}],
        "credit": {"weights": {"outcome": 0.4, "verifier": 0.4, "dependency": 0.2}},
        "repair": {"turn_threshold": 0.95, "subtrajectory_threshold": 0.95},
        "experience": {"graph_dir": "outputs/test/graph", "min_credit": 0.6},
        "training": {"model_name_or_path": "local-model"},
        "evaluation": {},
    }
    system = build_system(cfg, use_graph=False)
    sample = load_samples("examples/sample_dataset.jsonl")[0]

    class DummyRetriever:
        def retrieve(self, **kwargs):
            role = kwargs["role"]
            return [
                ExperienceNode(
                    node_id=f"node_{role}",
                    text=f"retrieved for {role}",
                    role=role,
                    task_type="qa",
                    credit=0.9,
                    source_episode_id="ep",
                    source_turn_ids=["t"],
                )
            ]

    system["runtime"].retriever = DummyRetriever()
    episode = system["runtime"].run_sample(
        sample,
        use_retrieval=True,
        extra_context={"retrieval_enabled_roles": ["solver"]},
    )
    planner_turn, solver_turn, verifier_turn, summarizer_turn = episode.turns
    assert planner_turn.meta["retrieved_node_ids"] == []
    assert solver_turn.meta["retrieved_node_ids"] == ["node_solver"]
    assert verifier_turn.meta["retrieved_node_ids"] == []
    assert summarizer_turn.meta["retrieved_node_ids"] == []
