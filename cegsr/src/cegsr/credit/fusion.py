from __future__ import annotations

from collections import defaultdict

from cegsr.trajectories.schema import CreditRecord, EpisodeTrajectory


def fuse_credit_records(
    episode: EpisodeTrajectory,
    credit_record_groups: list[list[CreditRecord]],
    weights: dict[str, float] | None = None,
) -> list[CreditRecord]:
    """Weighted fusion over heterogeneous credit signals."""
    weights = weights or {}
    merged: dict[tuple[str, str], dict[str, object]] = defaultdict(lambda: {"signals": {}, "weighted_sum": 0.0, "weight_total": 0.0, "details": {}})
    for group in credit_record_groups:
        for record in group:
            signal_items = record.signals.items() or [("unknown", record.total)]
            for signal_name, signal_value in signal_items:
                weight = weights.get(signal_name, 1.0)
                bucket = merged[(record.target_type, record.target_id)]
                bucket["signals"][signal_name] = signal_value
                bucket["weighted_sum"] = float(bucket["weighted_sum"]) + weight * signal_value
                bucket["weight_total"] = float(bucket["weight_total"]) + weight
                bucket["details"].update(record.details)
    fused: list[CreditRecord] = []
    for (target_type, target_id), bucket in merged.items():
        total = float(bucket["weighted_sum"]) / max(1e-8, float(bucket["weight_total"]))
        fused.append(
            CreditRecord(
                target_type=target_type,
                target_id=target_id,
                total=round(total, 6),
                signals={k: float(v) for k, v in dict(bucket["signals"]).items()},
                details=dict(bucket["details"]),
            )
        )
    episode.credit_records = sorted(fused, key=lambda x: (x.target_type, x.target_id))
    return episode.credit_records
