from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from cegsr.trajectories.schema import ExperienceEdge, ExperienceNode
from cegsr.utils.io import read_jsonl, write_jsonl


@dataclass
class GraphStore:
    root_dir: str
    nodes: dict[str, ExperienceNode] = field(default_factory=dict)
    edges: dict[str, ExperienceEdge] = field(default_factory=dict)

    @property
    def node_path(self) -> Path:
        return Path(self.root_dir) / "nodes.jsonl"

    @property
    def edge_path(self) -> Path:
        return Path(self.root_dir) / "edges.jsonl"

    def add_nodes(self, nodes: list[ExperienceNode]) -> None:
        for node in nodes:
            self.nodes[node.node_id] = node

    def add_edges(self, edges: list[ExperienceEdge]) -> None:
        for edge in edges:
            self.edges[edge.edge_id] = edge

    def neighbors(self, node_id: str) -> list[ExperienceNode]:
        out = []
        for edge in self.edges.values():
            if edge.source_id == node_id and edge.target_id in self.nodes:
                out.append(self.nodes[edge.target_id])
            elif edge.target_id == node_id and edge.source_id in self.nodes:
                out.append(self.nodes[edge.source_id])
        return out

    def save(self) -> None:
        write_jsonl(self.node_path, [n.to_dict() for n in self.nodes.values()])
        write_jsonl(self.edge_path, [e.to_dict() for e in self.edges.values()])

    @classmethod
    def load(cls, root_dir: str) -> "GraphStore":
        store = cls(root_dir=root_dir)
        for item in read_jsonl(store.node_path):
            node = ExperienceNode.from_dict(item)
            store.nodes[node.node_id] = node
        for item in read_jsonl(store.edge_path):
            edge = ExperienceEdge.from_dict(item)
            store.edges[edge.edge_id] = edge
        return store

    def stats(self) -> dict[str, int]:
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
        }
