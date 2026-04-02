from cegsr.trajectories.schema import EpisodeTrajectory, TaskSample, AgentTurn


def test_episode_roundtrip():
    episode = EpisodeTrajectory(
        episode_id="ep1",
        sample=TaskSample(sample_id="s1", question="1+1?", answer="2"),
        turns=[AgentTurn(turn_id="t1", role="solver", prompt_messages=[{"role":"user","content":"Q"}], response="2")],
    )
    clone = EpisodeTrajectory.from_dict(episode.to_dict())
    assert clone.sample.answer == "2"
    assert clone.turns[0].response == "2"
