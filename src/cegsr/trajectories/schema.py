from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TaskSample:
    sample_id: str
    question: str
    answer: str
    context: str = ""
    choices: list[str] = field(default_factory=list)
    task_type: str = "qa"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskSample":
        return cls(**data)


@dataclass
class AgentTurn:
    turn_id: str
    role: str
    prompt_messages: list[dict[str, str]]
    response: str
    dependencies: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentTurn":
        return cls(**data)


@dataclass
class SubTrajectory:
    sub_id: str
    turn_ids: list[str]
    roles: list[str]
    summary: str
    start_turn: int
    end_turn: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubTrajectory":
        return cls(**data)


@dataclass
class CreditRecord:
    target_type: str
    target_id: str
    total: float
    signals: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CreditRecord":
        return cls(**data)


@dataclass
class RepairRecord:
    repair_id: str
    target_type: str
    target_id: str
    old_span: list[dict[str, Any]]
    new_span: list[dict[str, Any]]
    why_repaired: str
    kept_context_turn_ids: list[str] = field(default_factory=list)
    verifier_before: float | None = None
    verifier_after: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RepairRecord":
        return cls(**data)


@dataclass
class ExperienceNode:
    node_id: str
    text: str
    role: str
    task_type: str
    credit: float
    source_episode_id: str
    source_turn_ids: list[str]
    embedding: list[float] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    is_repaired: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperienceNode":
        return cls(**data)


@dataclass
class ExperienceEdge:
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    weight: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperienceEdge":
        return cls(**data)


@dataclass
class EpisodeTrajectory:
    episode_id: str
    sample: TaskSample
    turns: list[AgentTurn]
    subtrajectories: list[SubTrajectory] = field(default_factory=list)
    final_prediction: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    credit_records: list[CreditRecord] = field(default_factory=list)
    repair_records: list[RepairRecord] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeTrajectory":
        return cls(
            episode_id=data["episode_id"],
            sample=TaskSample.from_dict(data["sample"]),
            turns=[AgentTurn.from_dict(x) for x in data.get("turns", [])],
            subtrajectories=[SubTrajectory.from_dict(x) for x in data.get("subtrajectories", [])],
            final_prediction=data.get("final_prediction", ""),
            metrics=data.get("metrics", {}),
            reward=data.get("reward", 0.0),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            latency_s=data.get("latency_s", 0.0),
            credit_records=[CreditRecord.from_dict(x) for x in data.get("credit_records", [])],
            repair_records=[RepairRecord.from_dict(x) for x in data.get("repair_records", [])],
            meta=data.get("meta", {}),
        )
