from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import time

from cegsr.agents.graph_runtime import AgentGraphRuntime
from cegsr.agents.roles import build_agents
from cegsr.backends.hf_local import HFLocalBackend
from cegsr.backends.mock_backend import MockBackend
from cegsr.backends.sglang_backend import SGLangBackend
from cegsr.backends.vllm_backend import VLLMBackend
from cegsr.config.loader import load_config
from cegsr.credit.dependency_credit import DependencyCreditSignal
from cegsr.credit.fusion import fuse_credit_records
from cegsr.credit.outcome_credit import OutcomeCreditSignal
from cegsr.credit.verifier_credit import VerifierCreditSignal
from cegsr.evaluation.evaluators import evaluate_episodes
from cegsr.experience.builder import build_experience_graph_from_episodes
from cegsr.experience.graph_store import GraphStore
from cegsr.experience.retriever import ExperienceRetriever, LocalEmbedder
from cegsr.repair.selective_repair import SelectiveRepairEngine
from cegsr.tasks.mmlu_style import MMLUStyleTask
from cegsr.tasks.pubmedqa_style import PubMedQAStyleTask
from cegsr.tasks.qa import QATask
from cegsr.trajectories.schema import CreditRecord, EpisodeTrajectory, TaskSample
from cegsr.trajectories.segmentation import segment_episode
from cegsr.training.credit_filter import CreditFilterConfig, filter_by_credit
from cegsr.training.exporters import (
    export_credit_guided_preference,
    export_credit_guided_sft,
    export_preference_pairs,
    export_reward_data,
    export_role_sft,
)
from cegsr.training.llamafactory_adapter import build_dataset_info, generate_llamafactory_project
from cegsr.training.trainer import train_dpo, train_role_sft
from cegsr.utils.io import ensure_dir, read_jsonl, write_csv, write_json, write_jsonl
from cegsr.utils.logging import get_logger
from cegsr.utils.modeling import resolve_local_model_path

logger = get_logger(__name__)


def _iter_with_progress(items: list[Any], desc: str):
    total = len(items)
    try:
        from tqdm.auto import tqdm  # type: ignore

        return tqdm(items, total=total, desc=desc, dynamic_ncols=True)
    except Exception:
        def _generator():
            started = time.time()
            step = max(1, total // 20) if total else 1
            for idx, item in enumerate(items, start=1):
                if idx == 1 or idx == total or idx % step == 0:
                    elapsed = time.time() - started
                    logger.info('%s %d/%d (%.1f%%, %.1fs elapsed)', desc, idx, total, 100.0 * idx / max(1, total), elapsed)
                yield item

        return _generator()


def _print_progress(message: str) -> None:
    print(message, flush=True)


def load_samples(dataset_path: str, prepare_config: str | None = None) -> list[TaskSample]:
    path = Path(dataset_path)
    if not path.exists():
        hint = (
            f'Run `python scripts/prepare_data.py --config {prepare_config}` first, '
            if prepare_config
            else 'Run `python scripts/prepare_data.py --config configs/datasets/reasoning_mix_eval.yaml` first, '
        )
        raise FileNotFoundError(
            f'Dataset file not found: {dataset_path}. {hint}or point task.dataset_path to an existing jsonl.'
        )
    return [TaskSample.from_dict(item) for item in read_jsonl(path)]


def load_episodes(path: str) -> list[EpisodeTrajectory]:
    return [EpisodeTrajectory.from_dict(item) for item in read_jsonl(path)]


def save_episodes(path: str, episodes: list[EpisodeTrajectory]) -> None:
    write_jsonl(path, [ep.to_dict() for ep in episodes])


def make_task(task_type: str):
    if task_type == 'qa':
        return QATask()
    if task_type == 'mmlu_style':
        return MMLUStyleTask()
    if task_type == 'pubmedqa_style':
        return PubMedQAStyleTask()
    raise ValueError(f'Unsupported task_type: {task_type}')


def make_backend(cfg: dict[str, Any], adapter_paths: dict[str, str] | None = None):
    kind = cfg.get('kind', 'mock')
    if kind == 'mock':
        return MockBackend()
    if kind == 'hf_local':
        return HFLocalBackend(
            model_name_or_path=cfg['model_name_or_path'],
            device=cfg.get('device', 'auto'),
            trust_remote_code=cfg.get('trust_remote_code', True),
            load_kwargs=cfg.get('load_kwargs', {}),
            use_chat_template=cfg.get('use_chat_template', True),
            model_size=cfg.get('model_size'),
            adapter_paths=adapter_paths,
        )
    if kind == 'vllm':
        model = resolve_local_model_path(
            cfg['model'],
            model_size_hint=cfg.get('model_size'),
        )
        return VLLMBackend(
            model=model,
            base_url=cfg['base_url'],
            api_key=cfg.get('api_key', 'EMPTY'),
            extra_body=cfg.get('extra_body', {}),
        )
    if kind == 'sglang':
        model = resolve_local_model_path(
            cfg['model'],
            model_size_hint=cfg.get('model_size'),
        )
        return SGLangBackend(
            model=model,
            base_url=cfg['base_url'],
            api_key=cfg.get('api_key', 'EMPTY'),
            extra_body=cfg.get('extra_body', {}),
        )
    raise ValueError(f'Unsupported backend kind: {kind}')


def build_system(config_or_path: str | dict[str, Any], use_graph: bool | None = None, adapter_paths: dict[str, str] | None = None):
    config = load_config(config_or_path) if isinstance(config_or_path, (str, Path)) else config_or_path
    task = make_task(config['task']['task_type'])
    backend = make_backend(config['backend'], adapter_paths=adapter_paths)
    agents = build_agents(config['agents'], backend)
    retriever = None
    if use_graph is None:
        use_graph = bool(config.get('method', {}).get('use_experience_graph', False))
    if use_graph:
        graph_dir = config.get('experience', {}).get('graph_dir')
        if graph_dir and Path(graph_dir).exists():
            store = GraphStore.load(graph_dir)
            experience_cfg = config.get('experience', {})
            retrieval_cfg = experience_cfg.get('retrieval', {})
            embedder = LocalEmbedder(model_name_or_path=experience_cfg.get('embed_model'))
            retriever = ExperienceRetriever(
                store,
                embedder,
                top_k=int(retrieval_cfg.get('top_k', 2)),
                expand_neighbors=bool(retrieval_cfg.get('expand_neighbors', False)),
                role_match_only=bool(retrieval_cfg.get('role_match_only', True)),
                exclude_same_sample=bool(retrieval_cfg.get('exclude_same_sample', True)),
                same_dataset_only=bool(retrieval_cfg.get('same_dataset_only', True)),
                min_similarity=float(retrieval_cfg.get('min_similarity', 0.3)),
                question_overlap_weight=float(retrieval_cfg.get('question_overlap_weight', 0.2)),
                min_question_overlap=float(retrieval_cfg.get('min_question_overlap', 0.0)),
            )
    runtime = AgentGraphRuntime(
        agents=agents,
        role_order=config['graph']['role_order'],
        task=task,
        retriever=retriever,
    )
    return {'config': config, 'task': task, 'backend': backend, 'runtime': runtime}


def collect_episodes(
    config_or_path: str | dict[str, Any],
    output_path: str | None = None,
    use_retrieval: bool = False,
    max_samples: int | None = None,
    verbose_turns: bool = False,
    adapter_paths: dict[str, str] | None = None,
) -> str:
    system = build_system(config_or_path, use_graph=use_retrieval, adapter_paths=adapter_paths)
    config = system['config']
    runtime = system['runtime']
    dataset_path = config['task']['dataset_path']
    samples = load_samples(dataset_path, prepare_config=config['task'].get('prepare_config'))
    if max_samples is not None:
        samples = samples[:max_samples]
    retrieval_cfg = config.get('experience', {}).get('retrieval', {})
    extra_context: dict[str, Any] = {'verbose_turns': verbose_turns}
    if use_retrieval:
        if 'top_k' in retrieval_cfg:
            extra_context['top_k'] = retrieval_cfg.get('top_k')
        enabled_roles = retrieval_cfg.get('enabled_roles')
        if enabled_roles:
            extra_context['retrieval_enabled_roles'] = enabled_roles
    output_path = output_path or str(Path(config['project']['output_dir']) / 'raw_episodes.jsonl')
    episodes = [
        runtime.run_sample(
            sample,
            use_retrieval=use_retrieval,
            extra_context=extra_context,
        )
        for sample in _iter_with_progress(samples, desc='Collect')
    ]
    save_episodes(output_path, episodes)
    return output_path


def annotate_episodes(
    episodes_path: str,
    config_or_path: str | dict[str, Any],
    output_path: str | None = None,
    trajectory_level_only: bool = False,
) -> str:
    config = load_config(config_or_path) if isinstance(config_or_path, (str, Path)) else config_or_path
    episodes = load_episodes(episodes_path)
    for episode in _iter_with_progress(episodes, desc='Credit'):
        segment_episode(
            episode,
            window_size=config['credit'].get('segment_window', 2),
            boundary_roles=tuple(config['credit'].get('segment_boundary_roles', ['verifier', 'summarizer'])),
        )
        if trajectory_level_only:
            score = float(episode.metrics.get('accuracy', 0))
            episode.credit_records = [
                CreditRecord(target_type='turn', target_id=turn.turn_id, total=score, signals={'trajectory': score})
                for turn in episode.turns
            ]
            episode.credit_records.extend(
                CreditRecord(target_type='subtrajectory', target_id=sub.sub_id, total=score, signals={'trajectory': score})
                for sub in episode.subtrajectories
            )
            roles = sorted({turn.role for turn in episode.turns})
            episode.credit_records.extend(
                CreditRecord(target_type='role', target_id=role, total=score, signals={'trajectory': score})
                for role in roles
            )
        else:
            groups = [
                OutcomeCreditSignal().compute(episode),
                VerifierCreditSignal().compute(episode),
                DependencyCreditSignal().compute(episode),
            ]
            fuse_credit_records(episode, groups, weights=config['credit'].get('weights', {}))
    output_path = output_path or str(Path(config['project']['output_dir']) / 'annotated_episodes.jsonl')
    save_episodes(output_path, episodes)
    return output_path


def repair_episodes(episodes_path: str, config_or_path: str | dict[str, Any], output_path: str | None = None, adapter_paths: dict[str, str] | None = None) -> str:
    system = build_system(config_or_path, use_graph=False, adapter_paths=adapter_paths)
    config = system['config']
    runtime = system['runtime']
    repairer = SelectiveRepairEngine(
        runtime,
        turn_threshold=config['repair'].get('turn_threshold', 0.45),
        subtrajectory_threshold=config['repair'].get('subtrajectory_threshold', 0.45),
        require_verifier_issue=config['repair'].get('require_verifier_issue', True),
        verifier_issue_threshold=config['repair'].get('verifier_issue_threshold', 0.6),
        relax_on_failure=config['repair'].get('relax_on_failure', True),
        failure_margin=config['repair'].get('failure_margin', 0.08),
    )
    episodes = load_episodes(episodes_path)
    repaired: list[EpisodeTrajectory] = []
    for episode in _iter_with_progress(episodes, desc='Repair'):
        repaired.append(repairer.repair(episode, use_retrieval=False))
    output_path = output_path or str(Path(config['project']['output_dir']) / 'repaired_episodes.jsonl')
    save_episodes(output_path, repaired)
    return output_path


def build_experience_graph(episodes_path: str, config_or_path: str | dict[str, Any], graph_dir: str | None = None) -> str:
    config = load_config(config_or_path) if isinstance(config_or_path, (str, Path)) else config_or_path
    graph_dir = graph_dir or config['experience']['graph_dir']
    episodes = load_episodes(episodes_path)
    build_experience_graph_from_episodes(
        episodes=episodes,
        graph_dir=graph_dir,
        min_credit=config['experience'].get('min_credit', 0.6),
        embed_model=config['experience'].get('embed_model'),
    )
    return graph_dir


def export_training_data(episodes_path: str, config_or_path: str | dict[str, Any], export_dir: str | None = None) -> dict[str, str]:
    config = load_config(config_or_path) if isinstance(config_or_path, (str, Path)) else config_or_path
    episodes = load_episodes(episodes_path)
    export_dir = export_dir or str(Path(config['project']['output_dir']) / 'training_data')
    manifest = export_role_sft(episodes, export_dir)
    export_preference_pairs(episodes, export_dir)
    export_reward_data(episodes, export_dir)
    generate_llamafactory_project(
        export_dir=export_dir,
        model_name_or_path=resolve_local_model_path(
            config['training']['model_name_or_path'],
            model_size_hint=config['training'].get('model_size'),
        ),
        output_dir=str(Path(config['project']['output_dir']) / 'llamafactory_runs'),
        lora_template=config['training'].get('lora_template', {}),
        qlora_template=config['training'].get('qlora_template', {}),
        distributed_config=config['training'].get('distributed'),
    )
    return manifest


def run_pipeline(config_or_path: str | dict[str, Any], output_dir: str | None = None) -> dict[str, Any]:
    config = load_config(config_or_path) if isinstance(config_or_path, (str, Path)) else deepcopy(config_or_path)
    if output_dir:
        config['project']['output_dir'] = output_dir
        config.setdefault('experience', {})['graph_dir'] = str(Path(output_dir) / 'graph')

    project_dir = Path(config['project']['output_dir'])
    raw_file = str(project_dir / 'raw.jsonl')
    annotated_file = str(project_dir / 'annotated.jsonl')
    repaired_file = str(project_dir / 'repaired.jsonl')
    graph_dir = str(project_dir / 'graph')
    export_dir = str(project_dir / 'training_data')
    eval_dir = str(project_dir / 'eval')

    collect_episodes(config, output_path=raw_file, use_retrieval=False)
    annotate_episodes(raw_file, config, output_path=annotated_file)
    repair_episodes(annotated_file, config, output_path=repaired_file)
    build_experience_graph(repaired_file, config, graph_dir=graph_dir)
    export_training_data(repaired_file, config, export_dir=export_dir)
    metrics = evaluate_episode_file(repaired_file, eval_dir, graph_dir=graph_dir)

    return {
        'raw': raw_file,
        'annotated': annotated_file,
        'repaired': repaired_file,
        'graph_dir': graph_dir,
        'export_dir': export_dir,
        'eval_dir': eval_dir,
        'metrics': metrics,
    }


def evaluate_episode_file(episodes_path: str, output_dir: str, graph_dir: str | None = None) -> dict:
    episodes = load_episodes(episodes_path)
    graph_stats = GraphStore.load(graph_dir).stats() if graph_dir and Path(graph_dir).exists() else None
    return evaluate_episodes(episodes, output_dir=output_dir, graph_stats=graph_stats)


def run_end_to_end_method(config: dict[str, Any], method_name: str, output_dir: str) -> tuple[str, dict]:
    method_dir = Path(output_dir) / method_name
    ensure_dir(method_dir)
    dataset_samples = load_samples(config['task']['dataset_path'], prepare_config=config['task'].get('prepare_config'))
    _print_progress(f'[Ablation] Start {method_name} ({len(dataset_samples)} samples)')

    if method_name == 'single_agent':
        system = build_system(config, use_graph=False)
        from cegsr.baselines.single_agent import SingleAgentBaseline

        baseline = SingleAgentBaseline(system['runtime'])
        episodes = [baseline.run(sample) for sample in _iter_with_progress(dataset_samples, desc=f'{method_name}:Run')]
        out_file = str(method_dir / 'episodes.jsonl')
        save_episodes(out_file, episodes)
        metrics = evaluate_episode_file(out_file, str(method_dir / 'report'))
        _print_progress(f'[Ablation] Done {method_name} accuracy={metrics.get("accuracy", 0)}')
        return out_file, metrics

    if method_name == 'static_multi_agent':
        system = build_system(config, use_graph=False)
        from cegsr.baselines.multi_agent_static import StaticMultiAgentBaseline

        baseline = StaticMultiAgentBaseline(system['runtime'])
        episodes = [baseline.run(sample) for sample in _iter_with_progress(dataset_samples, desc=f'{method_name}:Run')]
        out_file = str(method_dir / 'episodes.jsonl')
        save_episodes(out_file, episodes)
        metrics = evaluate_episode_file(out_file, str(method_dir / 'report'))
        _print_progress(f'[Ablation] Done {method_name} accuracy={metrics.get("accuracy", 0)}')
        return out_file, metrics

    if method_name == 'sirius_lite':
        system = build_system(config, use_graph=False)
        from cegsr.baselines.sirius_lite import SiriusLiteBaseline

        baseline = SiriusLiteBaseline(system['runtime'])
        episodes = [baseline.run(sample) for sample in _iter_with_progress(dataset_samples, desc=f'{method_name}:Run')]
        out_file = str(method_dir / 'episodes.jsonl')
        save_episodes(out_file, episodes)
        metrics = evaluate_episode_file(out_file, str(method_dir / 'report'))
        _print_progress(f'[Ablation] Done {method_name} accuracy={metrics.get("accuracy", 0)}')
        return out_file, metrics

    raw_file = str(method_dir / 'raw.jsonl')
    annotated_file = str(method_dir / 'annotated.jsonl')
    repaired_file = str(method_dir / 'repaired.jsonl')
    graph_dir = str(method_dir / 'graph')

    collect_episodes(config, output_path=raw_file, use_retrieval=False)
    if method_name == 'offline_sft_only':
        annotate_episodes(raw_file, config, output_path=annotated_file)
        export_training_data(annotated_file, config, export_dir=str(method_dir / 'export'))
        metrics = evaluate_episode_file(annotated_file, str(method_dir / 'report'))
        _print_progress(f'[Ablation] Done {method_name} accuracy={metrics.get("accuracy", 0)}')
        return annotated_file, metrics

    annotate_episodes(
        raw_file,
        config,
        output_path=annotated_file,
        trajectory_level_only=(method_name == 'trajectory_level_credit'),
    )

    if method_name in {'repair_only', 'ours_wo_graph', 'ours_full'}:
        repair_episodes(annotated_file, config, output_path=repaired_file)
    current_eval_path = repaired_file if Path(repaired_file).exists() else annotated_file

    if method_name in {'ours_full', 'ours_wo_selective_repair'}:
        build_experience_graph(current_eval_path if method_name == 'ours_full' else annotated_file, config, graph_dir=graph_dir)
        cfg_with_graph = deepcopy(config)
        cfg_with_graph['experience']['graph_dir'] = graph_dir
        cfg_with_graph.setdefault('method', {})['use_experience_graph'] = True
        collect_episodes(cfg_with_graph, output_path=str(method_dir / 'retrieved_eval.jsonl'), use_retrieval=True)
        current_eval_path = str(method_dir / 'retrieved_eval.jsonl')

    metrics = evaluate_episode_file(current_eval_path, str(method_dir / 'report'), graph_dir=graph_dir if Path(graph_dir).exists() else None)
    _print_progress(f'[Ablation] Done {method_name} accuracy={metrics.get("accuracy", 0)}')
    return current_eval_path, metrics


def run_ablation_suite(config_or_path: str | dict[str, Any], output_dir: str | None = None, methods: list[str] | None = None) -> dict[str, dict]:
    config = load_config(config_or_path) if isinstance(config_or_path, (str, Path)) else config_or_path
    output_dir = output_dir or str(Path(config['project']['output_dir']) / 'ablations')
    if methods is None:
        methods = config['evaluation'].get(
            'methods',
            [
                'single_agent',
                'static_multi_agent',
                'sirius_lite',
                'ours_wo_graph',
                'ours_wo_selective_repair',
                'trajectory_level_credit',
                'repair_only',
                'offline_sft_only',
                'ours_full',
            ],
        )
    results: dict[str, dict] = {}
    for method in methods:
        _, metrics = run_end_to_end_method(config, method, output_dir)
        results[method] = metrics
    rows = []
    for method, metrics in results.items():
        row = {'method': method}
        row.update(metrics)
        rows.append(row)

    write_csv(Path(output_dir) / 'ablation_table.csv', rows)
    write_json(Path(output_dir) / 'ablation_table.json', rows)
    md = ['# Ablation Table', '', '| method | accuracy | exact_match | repair_coverage | retrieval_proxy |', '|---|---:|---:|---:|---:|']
    for row in rows:
        md.append(
            f"| {row['method']} | {row.get('accuracy', 0)} | {row.get('exact_match', 0)} | {row.get('repair_coverage', 0)} | {row.get('retrieval_hit_usefulness_proxy', 0)} |"
        )
    (Path(output_dir) / 'ablation_table.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    return results


# ─── Training Loop & Iterative Self-Improvement ───────────────────────


def export_credit_guided_training_data(
    episodes_path: str,
    config: dict[str, Any],
    export_dir: str,
) -> dict[str, Any]:
    """Export credit-filtered training data for LLaMA-Factory.

    Returns dict with paths and filter statistics.
    """
    episodes = load_episodes(episodes_path)
    export_path = ensure_dir(export_dir)

    filter_cfg = config.get('training', {}).get('credit_filter', {})
    credit_cfg = CreditFilterConfig(
        selection_mode=filter_cfg.get('selection_mode', 'percentile'),
        top_percentile=filter_cfg.get('top_percentile', 50.0),
        high_threshold=filter_cfg.get('high_threshold', 0.65),
        low_threshold=filter_cfg.get('low_threshold', 0.35),
        include_repaired=filter_cfg.get('include_repaired', True),
        min_samples_per_role=filter_cfg.get('min_samples_per_role', 10),
    )

    filtered = filter_by_credit(episodes, credit_cfg)
    logger.info(
        'Credit filter: %d/%d turns selected (%.1f%%), %d preference pairs',
        filtered.stats['sft_selected'],
        filtered.stats['total_turns'],
        filtered.stats['selection_rate'] * 100,
        filtered.stats['preference_pairs'],
    )

    manifest = export_credit_guided_sft(filtered.sft_turns, export_dir)
    pref_path = None
    if filtered.preference_pairs:
        pref_path = export_credit_guided_preference(filtered.preference_pairs, export_dir)

    # Generate dataset_info.json for LLaMA-Factory
    dataset_info = build_dataset_info(manifest, preference_path=pref_path)
    write_json(export_path / 'dataset_info.json', dataset_info)

    write_json(export_path / 'filter_stats.json', filtered.stats)

    return {
        'manifest': manifest,
        'preference_path': pref_path,
        'stats': filtered.stats,
        'export_dir': str(export_path),
    }


def _free_gpu_memory() -> None:
    """Force garbage collection and release GPU memory.

    Calls gc twice with an empty_cache sandwich — a single gc pass often leaves
    tensor buffers alive because their owning objects hold weak refs cleared
    only on the next cycle. This matters before spawning a LLaMA-Factory
    training subprocess on a single-card box.
    """
    import gc
    for _ in range(2):
        gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            try:
                free_b, total_b = torch.cuda.mem_get_info()
                logger.info('GPU mem after release: free=%.1fGB / total=%.1fGB',
                            free_b / 1e9, total_b / 1e9)
            except Exception:
                pass
    except ImportError:
        pass


def _merge_export_dirs(export_dirs: list[str], merged_dir: str) -> dict[str, Any]:
    """Merge SFT and preference data from multiple iterations into one directory.

    Used for cross-iteration data accumulation so training benefits from all
    previous iterations' high-credit data, not just the current one.
    """
    merged = ensure_dir(merged_dir)
    by_role: dict[str, list[dict]] = {}
    pref_pairs: list[dict] = []

    for export_dir in export_dirs:
        export_path = Path(export_dir)
        manifest_file = export_path / 'sft_manifest.json'
        if not manifest_file.exists():
            continue
        import json as _json
        manifest = _json.loads(manifest_file.read_text(encoding='utf-8'))
        for role in manifest:
            sft_file = export_path / f'{role}_sft.jsonl'
            if sft_file.exists():
                by_role.setdefault(role, []).extend(read_jsonl(sft_file))
        pref_file = export_path / 'preference_pairs.jsonl'
        if pref_file.exists():
            pref_pairs.extend(read_jsonl(pref_file))

    new_manifest: dict[str, str] = {}
    for role, rows in by_role.items():
        path = merged / f'{role}_sft.jsonl'
        write_jsonl(path, rows)
        new_manifest[role] = str(path)
    write_json(merged / 'sft_manifest.json', new_manifest)

    pref_path = None
    if pref_pairs:
        pref_path = str(merged / 'preference_pairs.jsonl')
        write_jsonl(merged / 'preference_pairs.jsonl', pref_pairs)

    dataset_info = build_dataset_info(new_manifest, preference_path=pref_path)
    write_json(merged / 'dataset_info.json', dataset_info)

    merge_stats = {
        'num_iters_merged': len(export_dirs),
        'total_per_role': {r: len(rows) for r, rows in by_role.items()},
        'total_preference_pairs': len(pref_pairs),
    }
    logger.info('Cross-iter merge: %d iters → %s', len(export_dirs), merge_stats['total_per_role'])
    return merge_stats


def run_iterative_loop(
    config_or_path: str | dict[str, Any],
    output_dir: str | None = None,
    max_iterations: int | None = None,
    training_mode: str | None = None,
    dpo_enabled: bool | None = None,
    early_stop_patience: int = 2,
    credit_mode: str = 'fine_grained',
    cross_iteration_mix: bool | None = None,
) -> dict[str, Any]:
    """Run the full iterative self-improvement loop.

    Each iteration:
      1. Collect train-split trajectories (with current model)
      2. Credit assignment (+ selective repair if fine_grained)
      3. Export credit-guided training data
      4. Train via LLaMA-Factory (SFT + optional DPO)
      5. Evaluate on eval-split with fine-tuned model
         — reports BOTH raw and repaired accuracy

    Args:
        config_or_path: Config file path or dict.
        output_dir: Override output directory.
        max_iterations: Number of iterations (default from config).
        training_mode: 'lora' or 'qlora' (default from config).
        dpo_enabled: Whether to run DPO after SFT.
        early_stop_patience: Stop if no improvement for N iterations.
        credit_mode: 'fine_grained' (CEG-SR) or 'trajectory' (SiriuS-SFT baseline).
        cross_iteration_mix: Accumulate training data across iterations.

    Returns:
        Summary dict with per-iteration metrics and paths.
    """
    config = load_config(config_or_path) if isinstance(config_or_path, (str, Path)) else deepcopy(config_or_path)
    if output_dir:
        config['project']['output_dir'] = output_dir

    iter_cfg = config.get('training', {}).get('iterative', {})
    max_iter = max_iterations or iter_cfg.get('max_iterations', 3)
    mode = training_mode or iter_cfg.get('training_mode', 'qlora')
    do_dpo = dpo_enabled if dpo_enabled is not None else iter_cfg.get('dpo_enabled', False)
    do_cross_mix = cross_iteration_mix if cross_iteration_mix is not None else iter_cfg.get('cross_iteration_mix', True)
    train_dataset_path = iter_cfg.get('train_dataset_path', config['task']['dataset_path'])
    train_prepare_config = iter_cfg.get('train_dataset_config')
    eval_dataset_path = iter_cfg.get('eval_dataset_path', config['task']['dataset_path'])

    # In trajectory mode (SiriuS-SFT baseline): binary credit, no repair, no DPO
    is_trajectory_mode = credit_mode == 'trajectory'
    if is_trajectory_mode:
        do_dpo = False
        logger.info('SiriuS-SFT baseline mode: trajectory-level credit, no repair, no DPO')
        # Override credit filter to threshold mode with 0.5 (keep successful episodes)
        config_traj = deepcopy(config)
        config_traj.setdefault('training', {}).setdefault('credit_filter', {})
        config_traj['training']['credit_filter']['selection_mode'] = 'threshold'
        config_traj['training']['credit_filter']['high_threshold'] = 0.5

    base_dir = Path(config['project']['output_dir'])
    ensure_dir(base_dir)

    iteration_results: list[dict[str, Any]] = []
    best_accuracy = 0.0
    no_improve_count = 0
    current_adapters: dict[str, str] | None = None
    all_export_dirs: list[str] = []

    logger.info('='*60)
    logger.info('Starting iterative self-improvement loop')
    logger.info('  credit_mode: %s', credit_mode)
    logger.info('  max_iterations: %d', max_iter)
    logger.info('  training_mode: %s', mode)
    logger.info('  dpo_enabled: %s', do_dpo)
    logger.info('  cross_iteration_mix: %s', do_cross_mix)
    logger.info('  train_dataset: %s', train_dataset_path)
    logger.info('  eval_dataset: %s', eval_dataset_path)
    logger.info('='*60)

    # ── Baseline eval (no adapters, no training) ──
    logger.info('Running baseline evaluation (no adapters)...')
    baseline_dir = ensure_dir(base_dir / 'baseline')
    eval_config_base = deepcopy(config)
    eval_config_base['task']['dataset_path'] = eval_dataset_path
    baseline_raw_file = str(baseline_dir / 'eval_raw.jsonl')
    collect_episodes(eval_config_base, output_path=baseline_raw_file, use_retrieval=False, adapter_paths=None)
    _free_gpu_memory()
    baseline_raw_metrics = evaluate_episode_file(baseline_raw_file, str(baseline_dir / 'eval_raw_report'))
    baseline_result: dict[str, Any] = {
        'iteration': -1,
        'raw_accuracy': baseline_raw_metrics.get('accuracy', 0),
        'raw_metrics': baseline_raw_metrics,
    }
    if not is_trajectory_mode:
        baseline_ann = str(baseline_dir / 'eval_annotated.jsonl')
        baseline_rep = str(baseline_dir / 'eval_repaired.jsonl')
        annotate_episodes(baseline_raw_file, config, output_path=baseline_ann)
        repair_episodes(baseline_ann, eval_config_base, output_path=baseline_rep, adapter_paths=None)
        _free_gpu_memory()
        baseline_repaired_metrics = evaluate_episode_file(baseline_rep, str(baseline_dir / 'eval_repaired_report'))
        baseline_result['accuracy'] = baseline_repaired_metrics.get('accuracy', 0)
        baseline_result['repaired_metrics'] = baseline_repaired_metrics
    else:
        baseline_result['accuracy'] = baseline_result['raw_accuracy']

    logger.info('Baseline: raw_accuracy=%.4f, accuracy=%.4f',
                baseline_result['raw_accuracy'], baseline_result['accuracy'])
    write_json(str(baseline_dir / 'baseline_result.json'), baseline_result)

    for iteration in range(max_iter):
        iter_dir = ensure_dir(base_dir / f'iter_{iteration}')
        logger.info('')
        logger.info('━'*60)
        logger.info('ITERATION %d / %d', iteration, max_iter - 1)
        logger.info('━'*60)

        active_config = config_traj if is_trajectory_mode else config

        # ── Phase 1: Collect on train split ──
        logger.info('[Iter %d] Phase 1: Collecting train-split trajectories', iteration)
        train_config = deepcopy(active_config)
        train_config['task']['dataset_path'] = train_dataset_path
        if train_prepare_config:
            train_config['task']['prepare_config'] = train_prepare_config

        train_raw = str(iter_dir / 'train_raw.jsonl')
        collect_episodes(train_config, output_path=train_raw, use_retrieval=False, adapter_paths=current_adapters)
        _free_gpu_memory()

        # ── Phase 2: Credit + Repair ──
        train_annotated = str(iter_dir / 'train_annotated.jsonl')
        train_repaired = str(iter_dir / 'train_repaired.jsonl')

        if is_trajectory_mode:
            logger.info('[Iter %d] Phase 2: Trajectory-level credit (no repair)', iteration)
            annotate_episodes(train_raw, active_config, output_path=train_annotated, trajectory_level_only=True)
            train_repaired = train_annotated  # no repair
        else:
            logger.info('[Iter %d] Phase 2: Fine-grained credit + Selective repair', iteration)
            annotate_episodes(train_raw, active_config, output_path=train_annotated)
            repair_episodes(train_annotated, train_config, output_path=train_repaired, adapter_paths=current_adapters)
        _free_gpu_memory()

        # ── Phase 3: Export credit-guided training data ──
        logger.info('[Iter %d] Phase 3: Exporting credit-guided training data', iteration)
        export_dir = str(iter_dir / 'training_data')
        export_result = export_credit_guided_training_data(train_repaired, active_config, export_dir)
        all_export_dirs.append(export_dir)

        # ── Phase 3.5: Cross-iteration data merge ──
        if do_cross_mix and len(all_export_dirs) > 1:
            logger.info('[Iter %d] Phase 3.5: Merging %d iterations of training data', iteration, len(all_export_dirs))
            merged_dir = str(iter_dir / 'training_data_merged')
            merge_stats = _merge_export_dirs(all_export_dirs, merged_dir)
            training_dir = merged_dir
            export_result['merge_stats'] = merge_stats
        else:
            training_dir = export_dir

        # ── Phase 4: Train ──
        logger.info('[Iter %d] Phase 4: Training (mode=%s, data_dir=%s)', iteration, mode,
                     'merged' if training_dir != export_dir else 'current')
        adapters_dir = str(iter_dir / 'adapters')
        adapter_paths = train_role_sft(
            config=active_config,
            export_dir=training_dir,
            adapters_dir=adapters_dir,
            training_mode=mode,
        )

        if not adapter_paths:
            logger.error('[Iter %d] Training produced no adapters, stopping', iteration)
            break

        # Optional DPO — train on solver's adapter, then replace it
        if do_dpo and export_result.get('preference_path'):
            logger.info('[Iter %d] Phase 4b: DPO training on solver adapter', iteration)
            solver_adapter = adapter_paths.get('solver', next(iter(adapter_paths.values())))
            dpo_out = str(iter_dir / 'dpo_solver')
            dpo_result = train_dpo(active_config, training_dir, solver_adapter, dpo_out)
            if dpo_result:
                adapter_paths['solver'] = dpo_result
                logger.info('[Iter %d] Solver adapter updated to DPO: %s', iteration, dpo_result)

        current_adapters = adapter_paths

        # ── Phase 5: Evaluate on eval split ──
        logger.info('[Iter %d] Phase 5: Evaluating on eval split', iteration)
        eval_config = deepcopy(active_config)
        eval_config['task']['dataset_path'] = eval_dataset_path

        # 5a: Raw eval (model capability without repair)
        eval_raw = str(iter_dir / 'eval_raw.jsonl')
        collect_episodes(eval_config, output_path=eval_raw, use_retrieval=False, adapter_paths=current_adapters)
        _free_gpu_memory()

        raw_report_dir = str(iter_dir / 'eval_raw_report')
        raw_metrics = evaluate_episode_file(eval_raw, raw_report_dir)
        raw_accuracy = raw_metrics.get('accuracy', 0)

        # 5b: Repaired eval (system accuracy with repair)
        if not is_trajectory_mode:
            eval_annotated = str(iter_dir / 'eval_annotated.jsonl')
            eval_repaired = str(iter_dir / 'eval_repaired.jsonl')
            annotate_episodes(eval_raw, active_config, output_path=eval_annotated)
            repair_episodes(eval_annotated, eval_config, output_path=eval_repaired, adapter_paths=current_adapters)
            _free_gpu_memory()
            eval_report_dir = str(iter_dir / 'eval_report')
            repaired_metrics = evaluate_episode_file(eval_repaired, eval_report_dir)
            accuracy = repaired_metrics.get('accuracy', 0)
        else:
            repaired_metrics = raw_metrics
            accuracy = raw_accuracy

        logger.info('[Iter %d] raw_accuracy=%.4f, accuracy=%.4f (best=%.4f)',
                     iteration, raw_accuracy, accuracy, best_accuracy)

        iter_result = {
            'iteration': iteration,
            'accuracy': accuracy,
            'raw_accuracy': raw_accuracy,
            'raw_metrics': raw_metrics,
            'metrics': repaired_metrics,
            'adapter_paths': adapter_paths,
            'export_stats': export_result.get('stats', {}),
            'paths': {
                'train_raw': train_raw,
                'train_repaired': train_repaired,
                'export_dir': export_dir,
                'training_dir': training_dir,
                'adapters_dir': adapters_dir,
                'eval_raw': eval_raw,
                'eval_repaired': str(iter_dir / 'eval_repaired.jsonl') if not is_trajectory_mode else eval_raw,
            },
        }
        iteration_results.append(iter_result)
        write_json(str(iter_dir / 'iteration_result.json'), iter_result)

        # Early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            no_improve_count = 0
            write_json(str(base_dir / 'best_adapters.json'), adapter_paths)
        else:
            no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                logger.info('Early stopping: no improvement for %d iterations', no_improve_count)
                break

    # Save summary
    summary = {
        'total_iterations': len(iteration_results),
        'best_accuracy': best_accuracy,
        'credit_mode': credit_mode,
        'cross_iteration_mix': do_cross_mix,
        'baseline': baseline_result,
        'iteration_results': iteration_results,
    }
    write_json(str(base_dir / 'iterative_summary.json'), summary)

    # Write iteration curve as CSV (include raw accuracy)
    curve_rows = [
        {
            'iteration': r['iteration'],
            'raw_accuracy': r['raw_accuracy'],
            'accuracy': r['accuracy'],
            'sft_selected': r['export_stats'].get('sft_selected', 0),
            'preference_pairs': r['export_stats'].get('preference_pairs', 0),
            'selection_rate': r['export_stats'].get('selection_rate', 0),
            'effective_threshold': r['export_stats'].get('effective_threshold', 0),
        }
        for r in iteration_results
    ]
    write_csv(base_dir / 'iteration_curve.csv', curve_rows)

    logger.info('='*60)
    logger.info('Iterative loop complete: %d iterations, best accuracy=%.4f',
                len(iteration_results), best_accuracy)
    logger.info('  Baseline: raw=%.4f', baseline_result['raw_accuracy'])
    for r in iteration_results:
        logger.info('  Iter %d: raw=%.4f  repaired=%.4f  sft=%d  sel_rate=%.1f%%',
                     r['iteration'], r['raw_accuracy'], r['accuracy'],
                     r['export_stats'].get('sft_selected', 0),
                     r['export_stats'].get('selection_rate', 0) * 100)
    logger.info('='*60)

    return summary
