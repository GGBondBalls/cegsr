from __future__ import annotations

from abc import ABC, abstractmethod

from cegsr.trajectories.schema import CreditRecord, EpisodeTrajectory


class CreditSignal(ABC):
    name = "base"

    @abstractmethod
    def compute(self, episode: EpisodeTrajectory) -> list[CreditRecord]:
        raise NotImplementedError
