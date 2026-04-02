from tempfile import TemporaryDirectory

from cegsr.credit.outcome_credit import OutcomeCreditSignal
from cegsr.credit.verifier_credit import VerifierCreditSignal
from cegsr.credit.dependency_credit import DependencyCreditSignal
from cegsr.credit.fusion import fuse_credit_records
from cegsr.experience.builder import build_experience_graph_from_episodes
from cegsr.experience.graph_store import GraphStore
from cegsr.experience.retriever import ExperienceRetriever, LocalEmbedder, question_overlap
from cegsr.trajectories.schema import EpisodeTrajectory, TaskSample, AgentTurn, RepairRecord, ExperienceNode
from cegsr.trajectories.segmentation import segment_episode


def test_graph_build():
    ep = EpisodeTrajectory(
        episode_id="ep1",
        sample=TaskSample(sample_id="s1", question="2+2?", answer="4", metadata={"dataset_name": "toy"}),
        turns=[
            AgentTurn(turn_id="t1", role="planner", prompt_messages=[], response="plan"),
            AgentTurn(turn_id="t2", role="solver", prompt_messages=[], response="Proposed answer: 4", dependencies=["t1"]),
            AgentTurn(turn_id="t3", role="summarizer", prompt_messages=[], response="Final Answer: 4", dependencies=["t1","t2"]),
        ],
        metrics={"accuracy": 1, "exact_match": 1},
    )
    segment_episode(ep)
    groups = [OutcomeCreditSignal().compute(ep), VerifierCreditSignal().compute(ep), DependencyCreditSignal().compute(ep)]
    fuse_credit_records(ep, groups)
    with TemporaryDirectory() as tmp:
        store = build_experience_graph_from_episodes([ep], tmp, min_credit=0.1)
        loaded = GraphStore.load(tmp)
        assert store.stats()["num_nodes"] >= 1
        assert loaded.stats()["num_edges"] >= 1


def test_graph_builder_uses_repaired_turn_role():
    ep = EpisodeTrajectory(
        episode_id="ep_repair",
        sample=TaskSample(sample_id="sample_repair", question="2+2?", answer="4", metadata={"dataset_name": "toy"}),
        turns=[
            AgentTurn(turn_id="t1", role="planner", prompt_messages=[], response="plan"),
            AgentTurn(turn_id="t2", role="solver", prompt_messages=[], response="Answer: 5", dependencies=["t1"]),
            AgentTurn(turn_id="t3", role="summarizer", prompt_messages=[], response="Final Answer: 5", dependencies=["t1", "t2"]),
        ],
        metrics={"accuracy": 1, "exact_match": 1},
        repair_records=[
            RepairRecord(
                repair_id="r1",
                target_type="turn",
                target_id="t2",
                old_span=[{"response": "Answer: 5"}],
                new_span=[{"response": "Answer: 4"}],
                why_repaired="wrong arithmetic",
            )
        ],
    )
    segment_episode(ep)
    groups = [OutcomeCreditSignal().compute(ep), VerifierCreditSignal().compute(ep), DependencyCreditSignal().compute(ep)]
    fuse_credit_records(ep, groups)
    with TemporaryDirectory() as tmp:
        store = build_experience_graph_from_episodes([ep], tmp, min_credit=0.1)
        repaired_nodes = [node for node in store.nodes.values() if node.is_repaired]
        assert repaired_nodes
        assert repaired_nodes[0].role == "solver"
        assert repaired_nodes[0].meta["sample_id"] == "sample_repair"
        assert repaired_nodes[0].meta["dataset_name"] == "toy"


def test_retriever_excludes_same_sample_and_keeps_same_role_only():
    store = GraphStore(root_dir="unused")
    store.add_nodes(
        [
            ExperienceNode(
                node_id="same_sample",
                text="Answer: A. same sample leak",
                role="solver",
                task_type="qa",
                credit=0.95,
                source_episode_id="ep1",
                source_turn_ids=["t1"],
                meta={"sample_id": "s1", "dataset_name": "commonsense_qa"},
            ),
            ExperienceNode(
                node_id="other_role",
                text="Plan: think carefully.",
                role="planner",
                task_type="qa",
                credit=0.98,
                source_episode_id="ep2",
                source_turn_ids=["t2"],
                meta={"sample_id": "s2", "dataset_name": "commonsense_qa"},
            ),
            ExperienceNode(
                node_id="other_dataset",
                text="Answer: A. matching wording but wrong dataset family.",
                role="solver",
                task_type="qa",
                credit=0.99,
                source_episode_id="ep2b",
                source_turn_ids=["t2b"],
                meta={"sample_id": "s2b", "dataset_name": "ai2_arc"},
            ),
            ExperienceNode(
                node_id="usable",
                text="Answer: A. supported by matching evidence.",
                role="solver",
                task_type="qa",
                credit=0.9,
                source_episode_id="ep3",
                source_turn_ids=["t3"],
                meta={"sample_id": "s3", "dataset_name": "commonsense_qa"},
            ),
        ]
    )
    retriever = ExperienceRetriever(
        store,
        LocalEmbedder(),
        top_k=3,
        expand_neighbors=False,
        role_match_only=True,
        exclude_same_sample=True,
    )
    nodes = retriever.retrieve(
        role="solver",
        task_type="qa",
        query="Which answer is supported by matching evidence?",
        history=[],
        sample_id="s1",
        dataset_name="commonsense_qa",
    )
    assert [node.node_id for node in nodes] == ["usable"]


def test_question_overlap_prefers_related_questions():
    assert question_overlap("Where is a rug near the front door kept?", "Where would you keep a rug near your front door?") > 0
    assert question_overlap("Where is a rug near the front door kept?", "How do plants make food?") == 0
