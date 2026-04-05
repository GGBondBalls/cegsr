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
from cegsr.serving import create_server_manager
from cegsr.training.exporters import export_credit_guided_sft, export_preference_pairs, export_reward_data, export_role_sft
from cegsr.training.llamafactory_adapter import generate_llamafactory_project
from cegsr.training.registry import register_checkpoint, resolve_checkpoint
from cegsr.training.runner import run_training_pipeline
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


def make_backend(cfg: dict[str, Any]):
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


def build_system(config_or_path: str | dict[str, Any], use_graph: bool | None = None):
    config = load_config(config_or_path) if isinstance(config_or_path, (str, Path)) else config_or_path
    task = make_task(config['task']['task_type'])
    backend = make_backend(config['backend'])
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
    resume: bool = True,
) -> str:
    """Collect trajectories.  Saves incrementally so progress survives crashes."""
    from cegsr.backends.openai_compatible import ServerDownError

    system = build_system(config_or_path, use_graph=use_retrieval)
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

    # Resume: load already-processed episodes and skip those samples
    episodes: list[EpisodeTrajectory] = []
    done_ids: set[str] = set()
    partial_path = output_path + '.partial'
    if resume and Path(partial_path).exists():
        episodes = load_episodes(partial_path)
        done_ids = {ep.sample.sample_id for ep in episodes}
        logger.info('Resuming collect: %d/%d already done', len(done_ids), len(samples))
    remaining = [s for s in samples if s.sample_id not in done_ids]

    for sample in _iter_with_progress(remaining, desc='Collect'):
        try:
            ep = runtime.run_sample(
                sample,
                use_retrieval=use_retrieval,
                extra_context=extra_context,
            )
        except ServerDownError:
            # Server died — save what we have and re-raise so the caller
            # (run_iterative) can restart vLLM and call us again.
            logger.error('Server down during collect at sample %s, saving %d partial episodes',
                         sample.sample_id, len(episodes))
            save_episodes(partial_path, episodes)
            raise

        episodes.append(ep)

        # Incremental save every 50 samples
        if len(episodes) % 50 == 0:
            save_episodes(partial_path, episodes)

    save_episodes(output_path, episodes)
    # Clean up partial file
    if Path(partial_path).exists():
        Path(partial_path).unlink()
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


def repair_episodes(
    episodes_path: str,
    config_or_path: str | dict[str, Any],
    output_path: str | None = None,
    resume: bool = True,
) -> str:
    """Selective repair.  Saves incrementally so progress survives crashes."""
    from cegsr.backends.openai_compatible import ServerDownError

    system = build_system(config_or_path, use_graph=False)
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
    output_path = output_path or str(Path(config['project']['output_dir']) / 'repaired_episodes.jsonl')

    # Resume support
    repaired: list[EpisodeTrajectory] = []
    done_ids: set[str] = set()
    partial_path = output_path + '.partial'
    if resume and Path(partial_path).exists():
        repaired = load_episodes(partial_path)
        done_ids = {ep.episode_id for ep in repaired}
        logger.info('Resuming repair: %d/%d already done', len(done_ids), len(episodes))
    remaining = [ep for ep in episodes if ep.episode_id not in done_ids]

    for episode in _iter_with_progress(remaining, desc='Repair'):
        try:
            repaired.append(repairer.repair(episode, use_retrieval=False))
        except ServerDownError:
            logger.error('Server down during repair, saving %d partial episodes', len(repaired))
            save_episodes(partial_path, repaired)
            raise

        if len(repaired) % 50 == 0:
            save_episodes(partial_path, repaired)

    save_episodes(output_path, repaired)
    if Path(partial_path).exists():
        Path(partial_path).unlink()
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
    """Export ALL turns (no credit filtering). For credit-guided export, use export_credit_guided_training_data."""
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
        lora_template=config['training'].get('lora_template'),
        qlora_template=config['training'].get('qlora_template'),
        dpo_template=config['training'].get('dpo_template'),
        distributed_config=config['training'].get('distributed'),
    )
    return manifest


def export_credit_guided_training_data(
    episodes_path: str,
    config_or_path: str | dict[str, Any],
    export_dir: str | None = None,
    high_credit_threshold: float = 0.65,
) -> dict[str, str]:
    """Export training data with credit-guided filtering (only high-quality turns)."""
    config = load_config(config_or_path) if isinstance(config_or_path, (str, Path)) else config_or_path
    episodes = load_episodes(episodes_path)
    export_dir = export_dir or str(Path(config['project']['output_dir']) / 'training_data')

    threshold = config.get('training', {}).get('high_credit_threshold', high_credit_threshold)
    manifest = export_credit_guided_sft(episodes, export_dir, high_credit_threshold=threshold)
    export_preference_pairs(episodes, export_dir)
    export_reward_data(episodes, export_dir)

    generate_llamafactory_project(
        export_dir=export_dir,
        model_name_or_path=resolve_local_model_path(
            config['training']['model_name_or_path'],
            model_size_hint=config['training'].get('model_size'),
        ),
        output_dir=str(Path(config['project']['output_dir']) / 'llamafactory_runs'),
        lora_template=config['training'].get('lora_template'),
        qlora_template=config['training'].get('qlora_template'),
        dpo_template=config['training'].get('dpo_template'),
        distributed_config=config['training'].get('distributed'),
    )
    return manifest


def run_training(
    config_or_path: str | dict[str, Any],
    export_dir: str | None = None,
    training_mode: str | None = None,
    run_dpo: bool = False,
    merge_after: bool = True,
) -> dict[str, str]:
    """
    Execute LLaMA-Factory training from previously exported data.

    Returns dict with adapter/model paths.
    """
    config = load_config(config_or_path) if isinstance(config_or_path, (str, Path)) else config_or_path
    export_dir = export_dir or str(Path(config['project']['output_dir']) / 'training_data')
    training_cfg = config.get('training', {})
    mode = training_mode or training_cfg.get('mode', 'qlora')

    results = run_training_pipeline(
        export_dir=export_dir,
        training_mode=mode,
        run_dpo_stage=run_dpo,
        merge_after_training=merge_after,
        distributed_config=training_cfg.get('distributed'),
        timeout_per_stage=training_cfg.get('timeout_per_stage'),
    )

    # Register trained model in checkpoint registry
    registry_path = str(Path(config['project']['output_dir']) / 'checkpoint_registry.json')
    if 'merged_model' in results:
        register_checkpoint(registry_path, 'latest_merged', results['merged_model'], {'mode': mode})
    if 'sft_adapter' in results:
        register_checkpoint(registry_path, 'latest_sft', results['sft_adapter'], {'mode': mode})

    return results


def _run_with_server(
    fn,
    server,
    model_path: str | None,
    max_restarts: int = 3,
    **kwargs,
):
    """Call *fn* and auto-restart vLLM if it crashes mid-inference.

    *fn* must accept the same **kwargs and raise ServerDownError when the
    server is unreachable.  It should support ``resume=True`` so that
    progress already saved to the .partial file is not lost.
    """
    from cegsr.backends.openai_compatible import ServerDownError

    for attempt in range(1, max_restarts + 1):
        try:
            return fn(**kwargs)
        except ServerDownError:
            if attempt >= max_restarts:
                raise
            logger.warning(
                'Server down during %s (attempt %d/%d), restarting vLLM ...',
                fn.__name__, attempt, max_restarts,
            )
            if server:
                server.restart(model_path=model_path)


def _ensure_server(server, model_path: str | None) -> None:
    """Start or verify the vLLM server is healthy."""
    if server is None:
        return
    if not server.health_check():
        logger.info('vLLM not running, starting with model: %s', model_path)
        server.start(model_path=model_path)


def run_iterative(
    config_or_path: str | dict[str, Any],
    output_dir: str | None = None,
    max_iterations: int = 3,
    training_mode: str | None = None,
    run_dpo: bool = False,
    high_credit_threshold: float = 0.65,
    use_train_split: bool = True,
) -> dict[str, Any]:
    """
    Run the full iterative self-improvement loop.

    Each iteration follows the cycle::

        [vLLM ON]  collect → credit → repair → graph → export
        [vLLM OFF] train (SFT + optional DPO) → merge LoRA
        [vLLM ON]  eval (with the new merged model)

    vLLM is automatically stopped before training (to free GPU) and
    restarted afterwards with the merged model.  If vLLM crashes
    mid-inference, it is restarted and the step resumes from partial
    progress saved on disk.

    Returns per-iteration metrics and final results.
    """
    config = load_config(config_or_path) if isinstance(config_or_path, (str, Path)) else deepcopy(config_or_path)
    if output_dir:
        config['project']['output_dir'] = output_dir
    base_output = Path(config['project']['output_dir'])

    train_dataset = config.get('task', {}).get('train_dataset_path')
    eval_dataset = config['task']['dataset_path']
    if use_train_split and not train_dataset:
        logger.warning('No train_dataset_path configured; using eval dataset for both train and eval')
        train_dataset = eval_dataset

    mode = training_mode or config.get('training', {}).get('mode', 'qlora')
    server = create_server_manager(config)
    iteration_results: list[dict[str, Any]] = []

    try:
        for iteration in range(max_iterations):
            iter_dir = base_output / f'iteration_{iteration}'
            iter_config = deepcopy(config)
            iter_config['project']['output_dir'] = str(iter_dir)
            iter_config.setdefault('experience', {})['graph_dir'] = str(iter_dir / 'graph')

            logger.info('========== Iteration %d/%d ==========', iteration, max_iterations - 1)

            current_model = config.get('backend', {}).get('model')

            # ----------------------------------------------------------
            # Phase A: Inference (vLLM ON)
            # ----------------------------------------------------------
            _ensure_server(server, current_model)

            # Step 1: Collect on train split
            if use_train_split and train_dataset:
                iter_config['task']['dataset_path'] = train_dataset
            raw_file = str(iter_dir / 'raw.jsonl')
            _run_with_server(
                collect_episodes, server, current_model,
                config_or_path=iter_config, output_path=raw_file,
                use_retrieval=False, resume=True,
            )

            # Step 2: Credit assignment (CPU-only)
            annotated_file = str(iter_dir / 'annotated.jsonl')
            annotate_episodes(raw_file, iter_config, output_path=annotated_file)

            # Step 3: Selective repair (needs vLLM)
            repaired_file = str(iter_dir / 'repaired.jsonl')
            _run_with_server(
                repair_episodes, server, current_model,
                episodes_path=annotated_file, config_or_path=iter_config,
                output_path=repaired_file, resume=True,
            )

            # Step 4: Build experience graph (CPU-only)
            graph_dir = str(iter_dir / 'graph')
            build_experience_graph(repaired_file, iter_config, graph_dir=graph_dir)

            # Step 5: Export credit-guided training data (CPU-only)
            export_dir = str(iter_dir / 'training_data')
            export_credit_guided_training_data(
                repaired_file, iter_config, export_dir=export_dir,
                high_credit_threshold=high_credit_threshold,
            )

            # ----------------------------------------------------------
            # Phase B: Training (vLLM OFF, GPU for training)
            # ----------------------------------------------------------
            if server:
                logger.info('Stopping vLLM to free GPU for training ...')
                server.stop()

            train_results = run_training(
                iter_config, export_dir=export_dir,
                training_mode=mode, run_dpo=run_dpo, merge_after=True,
            )

            # Step 7: Swap model for next iteration
            merged_model = train_results.get('merged_model')
            if merged_model and Path(merged_model).exists():
                logger.info('Model updated: %s', merged_model)
                config['backend']['model'] = merged_model
                config['backend'].pop('model_size', None)
                config['training']['model_name_or_path'] = merged_model
                config['training'].pop('model_size', None)
                if 'serving' in config:
                    config['serving']['model_name_or_path'] = merged_model
                    config['serving'].pop('model_size', None)
            else:
                logger.warning('No merged model found, next iteration uses the same base model')

            # ----------------------------------------------------------
            # Phase C: Evaluation (vLLM ON with new model)
            # ----------------------------------------------------------
            new_model = config.get('backend', {}).get('model')
            if server:
                logger.info('Restarting vLLM with updated model: %s', new_model)
                server.start(model_path=new_model)

            eval_config = deepcopy(config)
            eval_config['task']['dataset_path'] = eval_dataset
            eval_config['project']['output_dir'] = str(iter_dir)

            eval_raw = str(iter_dir / 'eval_raw.jsonl')
            _run_with_server(
                collect_episodes, server, new_model,
                config_or_path=eval_config, output_path=eval_raw,
                use_retrieval=False, resume=True,
            )
            eval_annotated = str(iter_dir / 'eval_annotated.jsonl')
            annotate_episodes(eval_raw, eval_config, output_path=eval_annotated)
            eval_repaired = str(iter_dir / 'eval_repaired.jsonl')
            _run_with_server(
                repair_episodes, server, new_model,
                episodes_path=eval_annotated, config_or_path=eval_config,
                output_path=eval_repaired, resume=True,
            )
            eval_metrics = evaluate_episode_file(eval_repaired, str(iter_dir / 'eval'), graph_dir=graph_dir)

            iter_result = {
                'iteration': iteration,
                'train_results': train_results,
                'eval_metrics': eval_metrics,
                'model_path': merged_model or config.get('backend', {}).get('model', ''),
            }
            iteration_results.append(iter_result)
            write_json(str(iter_dir / 'iteration_result.json'), iter_result)
            logger.info(
                'Iteration %d complete: eval_accuracy=%.4f',
                iteration, eval_metrics.get('accuracy', 0),
            )

            if iteration >= 1:
                prev_acc = iteration_results[-2]['eval_metrics'].get('accuracy', 0)
                curr_acc = eval_metrics.get('accuracy', 0)
                if curr_acc <= prev_acc + 0.005:
                    logger.info('Accuracy saturated (%.4f → %.4f), stopping early', prev_acc, curr_acc)
                    break

    finally:
        if server and server.is_running:
            logger.info('Cleaning up vLLM ...')
            server.stop()

    summary = {
        'total_iterations': len(iteration_results),
        'iterations': iteration_results,
        'final_accuracy': iteration_results[-1]['eval_metrics'].get('accuracy', 0) if iteration_results else 0,
    }
    write_json(str(base_output / 'iterative_summary.json'), summary)
    logger.info('Iterative training complete: %d iterations, final_accuracy=%.4f',
                summary['total_iterations'], summary['final_accuracy'])
    return summary


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
