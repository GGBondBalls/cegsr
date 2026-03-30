from __future__ import annotations

from collections import defaultdict

from cegsr.experience.graph_store import GraphStore
from cegsr.experience.retriever import LocalEmbedder, cosine
from cegsr.trajectories.schema import EpisodeTrajectory, ExperienceEdge, ExperienceNode


def build_experience_graph_from_episodes(
    episodes: list[EpisodeTrajectory],
    graph_dir: str,
    min_credit: float = 0.6,
    embed_model: str | None = None,
) -> GraphStore:
    embedder = LocalEmbedder(model_name_or_path=embed_model)
    store = GraphStore(root_dir=graph_dir)
    role_nodes: dict[str, list[ExperienceNode]] = defaultdict(list)

    for episode in episodes:
        turn_credit = {r.target_id: r.total for r in episode.credit_records if r.target_type == "turn"}
        created_turn_node_ids: list[str] = []
        for turn in episode.turns:
            credit = turn_credit.get(turn.turn_id, 0.0)
            if credit < min_credit:
                continue
            node = ExperienceNode(
                node_id=f"node_{episode.episode_id}_{turn.turn_id}",
                text=turn.response,
                role=turn.role,
                task_type=episode.sample.task_type,
                credit=credit,
                source_episode_id=episode.episode_id,
                source_turn_ids=[turn.turn_id],
                embedding=embedder.encode(turn.response),
                tags=[turn.role, episode.sample.task_type],
                is_repaired=False,
                meta={
                    "sample_id": episode.sample.sample_id,
                    "dataset_name": episode.sample.metadata.get("dataset_name"),
                    "category": episode.sample.metadata.get("category"),
                },
            )
            store.add_nodes([node])
            role_nodes[turn.role].append(node)
            created_turn_node_ids.append(node.node_id)

        for i in range(len(created_turn_node_ids) - 1):
            store.add_edges(
                [
                    ExperienceEdge(
                        edge_id=f"edge_temporal_{episode.episode_id}_{i}",
                        source_id=created_turn_node_ids[i],
                        target_id=created_turn_node_ids[i + 1],
                        edge_type="temporal",
                        weight=1.0,
                    )
                ]
            )

        node_by_turn_id = {n.source_turn_ids[0]: n for n in store.nodes.values() if n.source_episode_id == episode.episode_id}
        for turn in episode.turns:
            if turn.turn_id not in node_by_turn_id:
                continue
            src = node_by_turn_id[turn.turn_id]
            for dep in turn.dependencies:
                if dep in node_by_turn_id:
                    tgt = node_by_turn_id[dep]
                    store.add_edges(
                        [
                            ExperienceEdge(
                                edge_id=f"edge_support_{src.node_id}_{tgt.node_id}",
                                source_id=tgt.node_id,
                                target_id=src.node_id,
                                edge_type="support",
                                weight=0.8,
                            )
                        ]
                    )

        if episode.repair_records:
            for idx, repair in enumerate(episode.repair_records):
                new_text = " ".join(item.get("response", "") for item in repair.new_span)
                repaired_role = next(
                    (turn.role for turn in episode.turns if turn.turn_id == repair.target_id),
                    episode.turns[0].role if episode.turns else "solver",
                )
                node = ExperienceNode(
                    node_id=f"node_repaired_{episode.episode_id}_{idx}",
                    text=new_text,
                    role=repaired_role,
                    task_type=episode.sample.task_type,
                    credit=episode.metrics.get("accuracy", 0.0),
                    source_episode_id=episode.episode_id,
                    source_turn_ids=[repair.target_id],
                    embedding=embedder.encode(new_text),
                    tags=["repaired", episode.sample.task_type],
                    is_repaired=True,
                    meta={
                        "sample_id": episode.sample.sample_id,
                        "dataset_name": episode.sample.metadata.get("dataset_name"),
                        "category": episode.sample.metadata.get("category"),
                    },
                )
                store.add_nodes([node])
                for old_node in list(store.nodes.values()):
                    if old_node.source_episode_id == episode.episode_id and repair.target_id in old_node.source_turn_ids:
                        store.add_edges(
                            [
                                ExperienceEdge(
                                    edge_id=f"edge_repaired_from_{old_node.node_id}_{node.node_id}",
                                    source_id=old_node.node_id,
                                    target_id=node.node_id,
                                    edge_type="repaired_from",
                                    weight=1.0,
                                )
                            ]
                        )

    for role, nodes in role_nodes.items():
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                score = cosine(nodes[i].embedding, nodes[j].embedding)
                if score >= 0.7:
                    store.add_edges(
                        [
                            ExperienceEdge(
                                edge_id=f"edge_same_role_{nodes[i].node_id}_{nodes[j].node_id}",
                                source_id=nodes[i].node_id,
                                target_id=nodes[j].node_id,
                                edge_type="same_role_pattern",
                                weight=score,
                            )
                        ]
                    )
                if score <= 0.2 and nodes[i].source_episode_id == nodes[j].source_episode_id:
                    store.add_edges(
                        [
                            ExperienceEdge(
                                edge_id=f"edge_contradiction_{nodes[i].node_id}_{nodes[j].node_id}",
                                source_id=nodes[i].node_id,
                                target_id=nodes[j].node_id,
                                edge_type="contradiction",
                                weight=1.0 - score,
                            )
                        ]
                    )

    store.save()
    return store
