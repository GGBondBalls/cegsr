from __future__ import annotations

from cegsr.evaluation.metrics import aggregate_metrics
from cegsr.evaluation.reports import write_run_report
from cegsr.trajectories.schema import EpisodeTrajectory


def evaluate_episodes(
    episodes: list[EpisodeTrajectory],
    output_dir: str,
    graph_stats: dict[str, int] | None = None,
) -> dict:
    metrics = aggregate_metrics(episodes, graph_stats=graph_stats)
    error_cases = [
        {
            'sample_id': ep.sample.sample_id,
            'dataset_name': ep.sample.metadata.get('dataset_name', 'unknown'),
            'pred': ep.final_prediction,
            'gold': ep.sample.answer,
            'question': ep.sample.question,
        }
        for ep in episodes
        if not ep.metrics.get('accuracy', 0)
    ][:20]
    write_run_report(output_dir, metrics, examples=error_cases)
    return metrics
