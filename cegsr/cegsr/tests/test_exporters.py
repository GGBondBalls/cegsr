from pathlib import Path
from tempfile import TemporaryDirectory

from cegsr.training.exporters import export_role_sft
from cegsr.trajectories.schema import EpisodeTrajectory, TaskSample, AgentTurn


def test_export_role_sft():
    episode = EpisodeTrajectory(
        episode_id="ep1",
        sample=TaskSample(sample_id="s1", question="1+1?", answer="2"),
        turns=[AgentTurn(turn_id="t1", role="solver", prompt_messages=[{"role":"user","content":"Q"}], response="2")],
    )
    with TemporaryDirectory() as tmp:
        manifest = export_role_sft([episode], tmp)
        assert "solver" in manifest
        assert Path(manifest["solver"]).exists()
