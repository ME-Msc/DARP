"""Shared type aliases for DARP planning models."""

# TODO(phase-4.3): Promote these aliases to public dataclasses if downstream
# integrations need runtime type metadata instead of hashable IDs.

from __future__ import annotations

from typing import Dict, Hashable, Mapping, Tuple

State = Hashable
Action = str
Observation = Hashable
Probability = float
Distribution = Dict[State, Probability]
ObservationDistribution = Dict[Observation, Probability]
TransitionKey = Tuple[State, Action, State]
ObservationKey = Tuple[Observation, State, Action]
ResetObservationKey = Tuple[Observation, State]
RewardKey = Tuple[State, Action]
DistributionLike = Mapping[State, Probability]
GroundAtom = Tuple[str, Tuple[str, ...]]
