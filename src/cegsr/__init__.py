"""CEG-SR research framework."""

from .workflows import (
    build_system,
    collect_episodes,
    annotate_episodes,
    repair_episodes,
    build_experience_graph,
    export_training_data,
    evaluate_episode_file,
    run_ablation_suite,
)

__all__ = [
    "build_system",
    "collect_episodes",
    "annotate_episodes",
    "repair_episodes",
    "build_experience_graph",
    "export_training_data",
    "evaluate_episode_file",
    "run_ablation_suite",
]

__version__ = "0.2.0"
