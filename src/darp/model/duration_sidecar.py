"""Load DurationModel definitions from JSON/YAML sidecars."""

# TODO(phase-9.1): Add optional risk/cost duration-sidecar fields for
# constrained benchmark experiments.

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

from darp.model.duration import (
    ActionName,
    DurationModel,
    FixedDurationModel,
    GaussianDurationModel,
    HistoryDurationEvaluator,
    StateDependentDurationModel,
)


class DurationSpecError(ValueError):
    """Raised when a duration sidecar is invalid. / duration sidecar 无效时抛出。"""


@dataclass(frozen=True)
class DurationSidecar:
    """Store a parsed duration sidecar and its model. / 保存解析后的 duration sidecar 及其模型。"""

    model: DurationModel
    raw: Mapping[str, Any]
    path: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def evaluator(self, horizon: float, zeta: float = 0.0) -> HistoryDurationEvaluator:
        """Build an evaluator using the RDDL horizon. / 使用 RDDL horizon 创建 evaluator。"""
        if horizon is None:
            raise DurationSpecError("DurationSidecar evaluator requires the horizon from the RDDL instance.")
        return HistoryDurationEvaluator(model=self.model, horizon=float(horizon), zeta=float(zeta))

    def validate_actions(self, action_names: set[str] | tuple[str, ...] | list[str]) -> None:
        """Validate that sidecar actions exist in the grounded model. / 验证 sidecar action 是否存在于 grounded model。"""
        available = {str(action) for action in action_names}
        unknown = sorted(action for action in duration_action_names(self.model) if action not in available)
        if unknown:
            raise DurationSpecError(f"Duration sidecar references unknown actions: {', '.join(unknown)}")


def load_duration_sidecar(path: str | Path) -> DurationSidecar:
    """Load a duration sidecar from JSON or YAML. / 从 JSON 或 YAML 加载 duration sidecar。"""
    sidecar_path = Path(path).expanduser()
    raw = _read_mapping(sidecar_path)
    return build_duration_sidecar(raw, path=sidecar_path)


def build_duration_sidecar(raw: Mapping[str, Any], path: str | Path | None = None) -> DurationSidecar:
    """Build a DurationSidecar from a parsed mapping. / 从解析后的 mapping 构建 DurationSidecar。"""
    config = _duration_config(raw)
    _validate_sidecar_schema(raw, config)
    model = build_duration_model(config)
    return DurationSidecar(
        model=model,
        raw=dict(raw),
        path=Path(path).expanduser() if path is not None else None,
        metadata={
            "kind": config.get("kind"),
            "source": "duration-sidecar",
        },
    )


def build_duration_model(config: Mapping[str, Any]) -> DurationModel:
    """Build a DurationModel from sidecar config. / 从 sidecar config 构建 DurationModel。"""
    kind = str(_required(config, "kind")).lower()
    if kind == "fixed":
        return FixedDurationModel(
            durations=_number_mapping(config.get("actions", {}), field_name="actions"),
            default=float(config.get("default", 1.0)),
        )
    if kind == "expected":
        return StateDependentDurationModel(
            durations=_state_action_numbers(config.get("state_actions", {})),
            default=float(config.get("default", 1.0)),
        )
    if kind == "gaussian":
        means, variances = _state_action_gaussians(config.get("state_actions", {}))
        return GaussianDurationModel(
            means=means,
            variances=variances,
            default_mean=float(config.get("default_mean", config.get("default", 1.0))),
            default_variance=float(config.get("default_variance", 0.0)),
        )
    raise DurationSpecError(f"Unsupported duration model kind: {kind}")


def duration_action_names(model: DurationModel) -> tuple[ActionName, ...]:
    """Return action names explicitly referenced by a duration model. / 返回 duration model 显式引用的 action 名称。"""
    actions: set[str] = set()
    if isinstance(model, FixedDurationModel):
        actions.update(str(action) for action in model.durations)
    elif isinstance(model, StateDependentDurationModel):
        actions.update(str(action) for _, action in model.durations)
    elif isinstance(model, GaussianDurationModel):
        actions.update(str(action) for _, action in model.means)
        actions.update(str(action) for _, action in model.variances)
    return tuple(sorted(actions))


def _duration_config(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the nested duration_model block when present. / 如有嵌套 duration_model 则返回该块。"""
    block = raw.get("duration_model", raw)
    if not isinstance(block, Mapping):
        raise DurationSpecError("duration_model must be a mapping.")
    return block


def _validate_sidecar_schema(raw: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Reject fields that belong to RDDL or old metadata. / 拒绝属于 RDDL 或旧 metadata 的字段。"""
    if "version" in raw:
        raise DurationSpecError("Duration sidecar no longer uses a version field; remove `version`.")
    if "horizon" in raw or "horizon" in config:
        raise DurationSpecError("Duration sidecar must not define horizon; use the RDDL instance horizon.")


def _read_mapping(path: Path) -> Mapping[str, Any]:
    """Read JSON/YAML sidecar content into a mapping. / 将 JSON/YAML sidecar 读为 mapping。"""
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        value = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        value = _loads_yaml_mapping(text)
    else:
        raise DurationSpecError(f"Unsupported duration sidecar extension: {path.suffix}")
    if not isinstance(value, Mapping):
        raise DurationSpecError("Duration sidecar root must be a mapping.")
    return value


def _loads_yaml_mapping(text: str) -> Mapping[str, Any]:
    """Load YAML using PyYAML when available, otherwise a small mapping-only parser. / 优先用 PyYAML，否则使用小型 mapping-only parser。"""
    try:
        import yaml  # type: ignore
    except ImportError:
        return _loads_simple_yaml_mapping(text)
    value = yaml.safe_load(text)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DurationSpecError("Duration YAML root must be a mapping.")
    return value


def _loads_simple_yaml_mapping(text: str) -> Mapping[str, Any]:
    """Parse the small YAML subset used by DARP sidecars. / 解析 DARP sidecar 使用的小型 YAML 子集。"""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_number, original_line in enumerate(text.splitlines(), start=1):
        line = _strip_yaml_comment(original_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            raise DurationSpecError(f"YAML indentation must use multiples of two spaces at line {line_number}.")
        key_value = line.strip()
        key, separator, value = key_value.partition(":")
        if not separator:
            raise DurationSpecError(f"YAML line {line_number} must contain ':'")
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        clean_key = key.strip()
        if not clean_key:
            raise DurationSpecError(f"YAML line {line_number} has an empty key.")
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[clean_key] = child
            stack.append((indent, child))
        else:
            parent[clean_key] = _parse_scalar(value.strip())
    return root


def _strip_yaml_comment(line: str) -> str:
    """Strip simple YAML comments outside quoted strings. / 去掉未加引号位置的简单 YAML 注释。"""
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index]
    return line


def _parse_scalar(value: str) -> Any:
    """Parse one YAML scalar. / 解析一个 YAML scalar。"""
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _number_mapping(value: Any, *, field_name: str) -> dict[str, float]:
    """Parse a mapping from name to numeric duration. / 解析 name 到数字 duration 的 mapping。"""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DurationSpecError(f"{field_name} must be a mapping.")
    return {str(key): float(item) for key, item in value.items()}


def _state_action_numbers(value: Any) -> dict[tuple[str, str], float]:
    """Parse state/action duration numbers. / 解析 state/action duration 数字。"""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DurationSpecError("state_actions must be a mapping.")
    result: dict[tuple[str, str], float] = {}
    for state, actions in value.items():
        if not isinstance(actions, Mapping):
            raise DurationSpecError(f"state_actions[{state!r}] must be a mapping.")
        for action, duration in actions.items():
            result[(str(state), str(action))] = float(duration)
    return result


def _state_action_gaussians(value: Any) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """Parse state/action Gaussian mean and variance entries. / 解析 state/action Gaussian 均值与方差。"""
    if value is None:
        return {}, {}
    if not isinstance(value, Mapping):
        raise DurationSpecError("state_actions must be a mapping.")
    means: dict[tuple[str, str], float] = {}
    variances: dict[tuple[str, str], float] = {}
    for state, actions in value.items():
        if not isinstance(actions, Mapping):
            raise DurationSpecError(f"state_actions[{state!r}] must be a mapping.")
        for action, entry in actions.items():
            key = (str(state), str(action))
            if isinstance(entry, Mapping):
                means[key] = float(entry.get("mean", entry.get("duration", 1.0)))
                variances[key] = float(entry.get("variance", 0.0))
            else:
                means[key] = float(entry)
                variances[key] = 0.0
    return means, variances


def _required(config: Mapping[str, Any], key: str) -> Any:
    """Return a required config value. / 返回必填 config 值。"""
    if key not in config:
        raise DurationSpecError(f"Duration sidecar is missing required field: {key}")
    return config[key]

