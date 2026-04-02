from cegsr.credit.outcome_credit import OutcomeCreditSignal
from cegsr.credit.verifier_credit import VerifierCreditSignal
from cegsr.credit.dependency_credit import DependencyCreditSignal
from cegsr.credit.fusion import fuse_credit_records
from cegsr.trajectories.schema import EpisodeTrajectory, TaskSample, AgentTurn
from cegsr.trajectories.segmentation import segment_episode


def build_episode():
    ep = EpisodeTrajectory(
        episode_id="ep1",
        sample=TaskSample(sample_id="s1", question="2+2?", answer="4"),
        turns=[
            AgentTurn(turn_id="t1", role="planner", prompt_messages=[], response="Plan 4"),
            AgentTurn(turn_id="t2", role="solver", prompt_messages=[], response="Proposed answer: 4", dependencies=["t1"]),
            AgentTurn(turn_id="t3", role="verifier", prompt_messages=[], response="VERDICT: correct\nScore: 0.9", dependencies=["t2"]),
        ],
        metrics={"accuracy": 1, "exact_match": 1},
    )
    segment_episode(ep)
    return ep


def test_fused_credit():
    ep = build_episode()
    groups = [OutcomeCreditSignal().compute(ep), VerifierCreditSignal().compute(ep), DependencyCreditSignal().compute(ep)]
    fused = fuse_credit_records(ep, groups, weights={"outcome": 0.4, "verifier": 0.4, "dependency": 0.2})
    assert any(r.target_type == "turn" for r in fused)
    assert any(r.target_type == "role" for r in fused)


def test_verifier_credit_does_not_copy_full_confidence_to_all_turns():
    ep = build_episode()
    records = VerifierCreditSignal().compute(ep)
    turn_scores = {r.target_id: r.total for r in records if r.target_type == "turn"}
    assert turn_scores["t3"] <= 0.9
    assert turn_scores["t1"] < turn_scores["t3"]
    assert turn_scores["t2"] < turn_scores["t3"]
