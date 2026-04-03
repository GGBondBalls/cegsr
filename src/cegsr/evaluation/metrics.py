from __future__ import annotations

from collections import Counter, defaultdict

from cegsr.trajectories.schema import EpisodeTrajectory


def aggregate_metrics(
    episodes: list[EpisodeTrajectory],
    graph_stats: dict[str, int] | None = None,
    training_manifest: dict[str, str] | None = None,
) -> dict[str, float | int]:
    graph_stats = graph_stats or {}
    total = max(1, len(episodes))
    acc = sum(float(ep.metrics.get('accuracy', 0)) for ep in episodes) / total
    em = sum(float(ep.metrics.get('exact_match', 0)) for ep in episodes) / total
    mcq = sum(float(ep.metrics.get('mcq_accuracy', 0)) for ep in episodes) / total
    avg_len = sum(len(ep.turns) for ep in episodes) / total
    avg_input_tokens = sum(ep.input_tokens for ep in episodes) / total
    avg_output_tokens = sum(ep.output_tokens for ep in episodes) / total

    repaired_episodes = [ep for ep in episodes if ep.repair_records]
    repair_coverage = len(repaired_episodes) / total
    repair_success_rate = (
        sum(float(ep.metrics.get('accuracy', 0)) for ep in repaired_episodes) / max(1, len(repaired_episodes))
    )
    changed_repairs = sum(1 for ep in repaired_episodes for rr in ep.repair_records if rr.meta.get('changed'))

    retrieval_usefulness = 0.0
    retrieved_turns = 0
    for ep in episodes:
        for turn in ep.turns:
            if turn.meta.get('retrieved_node_ids'):
                retrieved_turns += 1
                if ep.metrics.get('accuracy', 0):
                    retrieval_usefulness += 1.0
    retrieval_usefulness = retrieval_usefulness / max(1, retrieved_turns)

    role_counter = Counter()
    dataset_buckets: dict[str, list[float]] = defaultdict(list)
    category_buckets: dict[str, list[float]] = defaultdict(list)
    for ep in episodes:
        ds = ep.sample.metadata.get('dataset_name', 'unknown')
        dataset_buckets[ds].append(float(ep.metrics.get('accuracy', 0)))
        cat = ep.sample.metadata.get('category')
        if cat:
            category_buckets[str(cat)].append(float(ep.metrics.get('accuracy', 0)))
        for turn in ep.turns:
            role_counter[turn.role] += 1

    result: dict[str, float | int] = {
        'num_episodes': len(episodes),
        'accuracy': round(acc, 4),
        'exact_match': round(em, 4),
        'mcq_accuracy': round(mcq, 4),
        'repair_coverage': round(repair_coverage, 4),
        'repair_success_rate': round(repair_success_rate, 4),
        'num_changed_repairs': changed_repairs,
        'average_trajectory_length': round(avg_len, 4),
        'average_input_tokens': round(avg_input_tokens, 4),
        'average_output_tokens': round(avg_output_tokens, 4),
        'retrieval_hit_usefulness_proxy': round(retrieval_usefulness, 4),
        'graph_num_nodes': graph_stats.get('num_nodes', 0),
        'graph_num_edges': graph_stats.get('num_edges', 0),
    }
    for role, count in role_counter.items():
        result[f'training_data_size_by_role::{role}'] = count
    for dataset_name, values in dataset_buckets.items():
        result[f'dataset_accuracy::{dataset_name}'] = round(sum(values) / max(1, len(values)), 4)
        result[f'dataset_count::{dataset_name}'] = len(values)
    for category, values in category_buckets.items():
        result[f'category_accuracy::{category}'] = round(sum(values) / max(1, len(values)), 4)
    return result
