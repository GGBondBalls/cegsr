from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from cegsr.experience.graph_store import GraphStore
from cegsr.trajectories.schema import AgentTurn, ExperienceNode


class LocalEmbedder:
    """
    Graceful local embedding layer.
    - preferred: sentence-transformers
    - fallback: hashed bag-of-words
    """

    def __init__(self, model_name_or_path: str | None = None, dim: int = 128) -> None:
        self.model_name_or_path = model_name_or_path
        self.dim = dim
        self._model = None
        if model_name_or_path:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._model = SentenceTransformer(model_name_or_path)
            except Exception:
                self._model = None

    def encode(self, text: str) -> list[float]:
        if self._model is not None:
            vector = self._model.encode([text], normalize_embeddings=True)[0]
            return [float(x) for x in vector]
        tokens = re.findall(r"\w+", text.lower())
        counter = Counter(tokens)
        vector = [0.0] * self.dim
        for token, count in counter.items():
            idx = hash(token) % self.dim
            vector[idx] += float(count)
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    size = min(len(a), len(b))
    return sum(a[i] * b[i] for i in range(size))


def _node_embedding_text(node: ExperienceNode) -> str:
    source_question = str(node.meta.get("source_question", "")).strip()
    return f"role={node.role}\nquestion={source_question}\nresponse={node.text}"


class ExperienceRetriever:
    def __init__(
        self,
        graph_store: GraphStore,
        embedder: LocalEmbedder,
        *,
        top_k: int = 2,
        expand_neighbors: bool = False,
        role_match_only: bool = True,
        exclude_same_sample: bool = True,
        same_dataset_only: bool = True,
        min_similarity: float = 0.3,
    ) -> None:
        self.graph_store = graph_store
        self.embedder = embedder
        self.top_k = top_k
        self.expand_neighbors = expand_neighbors
        self.role_match_only = role_match_only
        self.exclude_same_sample = exclude_same_sample
        self.same_dataset_only = same_dataset_only
        self.min_similarity = min_similarity

    def retrieve(
        self,
        role: str,
        task_type: str,
        query: str,
        history: list[AgentTurn],
        sample_id: str | None = None,
        dataset_name: str | None = None,
        top_k: int | None = None,
        expand_neighbors: bool | None = None,
        role_match_only: bool | None = None,
        exclude_same_sample: bool | None = None,
        same_dataset_only: bool | None = None,
        min_similarity: float | None = None,
    ) -> list[ExperienceNode]:
        top_k = self.top_k if top_k is None else top_k
        expand_neighbors = self.expand_neighbors if expand_neighbors is None else expand_neighbors
        role_match_only = self.role_match_only if role_match_only is None else role_match_only
        exclude_same_sample = self.exclude_same_sample if exclude_same_sample is None else exclude_same_sample
        same_dataset_only = self.same_dataset_only if same_dataset_only is None else same_dataset_only
        min_similarity = self.min_similarity if min_similarity is None else min_similarity
        query_text = f"role={role}\ntask={task_type}\nquestion={query}\nhistory={' '.join(t.response for t in history[-2:])}"
        q = self.embedder.encode(query_text)
        candidates = []
        for node in self.graph_store.nodes.values():
            if node.task_type != task_type:
                continue
            if exclude_same_sample and sample_id is not None and node.meta.get("sample_id") == sample_id:
                continue
            if same_dataset_only and dataset_name and node.meta.get("dataset_name") not in {None, dataset_name}:
                continue
            if role_match_only:
                if node.role != role:
                    continue
            elif node.role not in {role, "summarizer", "solver"}:
                continue
            if not node.embedding:
                node.embedding = self.embedder.encode(_node_embedding_text(node))
            similarity = cosine(q, node.embedding)
            if similarity < min_similarity:
                continue
            score = similarity + 0.1 * float(node.credit)
            candidates.append((score, node))
        top = [node for _, node in sorted(candidates, key=lambda x: x[0], reverse=True)[:top_k]]
        if not expand_neighbors:
            return top
        expanded: dict[str, ExperienceNode] = {n.node_id: n for n in top}
        for node in top:
            for neighbor in self.graph_store.neighbors(node.node_id):
                expanded.setdefault(neighbor.node_id, neighbor)
        return list(expanded.values())[: max(top_k, len(top))]
