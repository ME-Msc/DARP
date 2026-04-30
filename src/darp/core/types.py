"""Shared type aliases for DARP planning models."""

# TODO(phase-4.1): Replace string-only aliases with richer typed identifiers if
# RDDL grounding needs object-typed variables.

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
RewardKey = Tuple[State, Action]
DistributionLike = Mapping[State, Probability]
