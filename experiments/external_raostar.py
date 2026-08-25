"""Validate and invoke the pinned external RAO* reimplementation.

The upstream checkout is intentionally external because its public repository
does not provide a redistribution license.  This module never imports upstream
code into the DARP process; a version-locked interpreter runs the small worker.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
from typing import Any, Mapping, Sequence

from experiments.external_raostar_worker import RESULT_PREFIX


DEFAULT_MANIFEST = Path(__file__).with_name("manifests") / "raostar_quad.json"
WORKER = Path(__file__).with_name("external_raostar_worker.py")


class ExternalRAOStarError(RuntimeError):
    """Raised when provenance or the external process cannot be trusted."""


@dataclass(frozen=True)
class CheckoutProvenance:
    checkout: str
    repository_url: str
    expected_commit: str
    actual_commit: str
    tracked_files_clean: bool
    required_files_present: bool
    license_status: str
    untracked_files_absent: bool = True
    required_files_match_head: bool = True

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load and minimally validate the version-controlled integration manifest."""
    resolved = Path(path).resolve()
    try:
        manifest = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExternalRAOStarError(f"Cannot load RAO* manifest {resolved}: {error}") from error
    required = {"schema_version", "algorithm", "license", "tested_environment", "required_files", "scenarios"}
    missing = sorted(required.difference(manifest))
    if missing:
        raise ExternalRAOStarError(f"RAO* manifest is missing fields: {missing}")
    if manifest["schema_version"] != 1:
        raise ExternalRAOStarError(
            f"Unsupported RAO* manifest schema: {manifest['schema_version']!r}"
        )
    if tuple(manifest["scenarios"]) != ("quad",):
        raise ExternalRAOStarError(
            "The external RAO* integration only supports the pinned Quad scenario."
        )
    if manifest["scenarios"]["quad"].get("adapter", {}).get("kind") != "upstream-quad":
        raise ExternalRAOStarError(
            "The Quad manifest must use the unchanged upstream Quad adapter."
        )
    return manifest


def _git(checkout: Path, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=15.0,
    )


def validate_checkout(
    checkout: Path,
    manifest: Mapping[str, Any],
) -> CheckoutProvenance:
    """Fail closed unless the external tree is the exact clean pinned commit."""
    root = Path(checkout).resolve()
    if not root.is_dir():
        raise ExternalRAOStarError(f"RAO* checkout does not exist: {root}")
    top_level = _git(root, ["rev-parse", "--show-toplevel"])
    if top_level.returncode != 0:
        raise ExternalRAOStarError(
            f"RAO* checkout is not a readable Git worktree: {top_level.stderr.strip()}"
        )
    actual_top_level = Path(top_level.stdout.strip()).resolve()
    if actual_top_level != root:
        raise ExternalRAOStarError(
            f"RAO* checkout must be the Git top-level directory: {root} != {actual_top_level}"
        )
    revision = _git(root, ["rev-parse", "HEAD"])
    if revision.returncode != 0:
        raise ExternalRAOStarError(
            f"RAO* checkout is not a readable Git worktree: {revision.stderr.strip()}"
        )
    actual_commit = revision.stdout.strip()
    expected_commit = str(manifest["algorithm"]["commit"])
    if actual_commit != expected_commit:
        raise ExternalRAOStarError(
            f"RAO* commit mismatch: expected {expected_commit}, found {actual_commit}"
        )
    status = _git(root, ["status", "--porcelain=v1", "--untracked-files=all"])
    if status.returncode != 0:
        raise ExternalRAOStarError(
            f"Could not verify the RAO* worktree: {status.stderr.strip()}"
        )
    if status.stdout.strip():
        raise ExternalRAOStarError(
            "RAO* checkout is not pristine (tracked modifications or untracked files); "
            "upstream algorithms, models, and import resolution must remain unchanged."
        )
    invalid: list[str] = []
    mismatched: list[str] = []
    for raw_relative in manifest["required_files"]:
        relative = str(raw_relative)
        relative_path = Path(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative
        ):
            invalid.append(relative)
            continue
        head_blob = _git(root, ["rev-parse", f"HEAD:{relative}"])
        worktree_blob = _git(
            root,
            ["hash-object", f"--path={relative}", "--", relative],
        )
        if (
            head_blob.returncode != 0
            or worktree_blob.returncode != 0
            or head_blob.stdout.strip() != worktree_blob.stdout.strip()
            or not (root / relative).is_file()
        ):
            mismatched.append(relative)
    if invalid:
        raise ExternalRAOStarError(
            f"RAO* manifest contains invalid required-file paths: {invalid}"
        )
    if mismatched:
        raise ExternalRAOStarError(
            "RAO* required files are absent or do not match their HEAD blobs: "
            f"{mismatched}"
        )
    return CheckoutProvenance(
        checkout=str(root),
        repository_url=str(manifest["algorithm"]["repository_url"]),
        expected_commit=expected_commit,
        actual_commit=actual_commit,
        tracked_files_clean=True,
        required_files_present=True,
        license_status=str(manifest["license"]["status"]),
        untracked_files_absent=True,
        required_files_match_head=True,
    )


def _archive_pinned_checkout(checkout: Path, commit: str, destination: Path) -> None:
    """Materialize the immutable pinned tree used by the worker."""

    completed = subprocess.run(
        ["git", "-C", str(checkout), "archive", "--format=tar", commit],
        check=False,
        capture_output=True,
        timeout=30.0,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ExternalRAOStarError(f"Could not archive pinned RAO* commit: {detail}")
    try:
        with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:") as archive:
            archive.extractall(destination, filter="data")
    except (tarfile.TarError, OSError) as error:
        raise ExternalRAOStarError(
            f"Could not materialize pinned RAO* source snapshot: {error}"
        ) from error


def _extract_worker_result(stdout: str) -> dict[str, Any]:
    payloads = [
        line[len(RESULT_PREFIX) :]
        for line in stdout.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    if len(payloads) != 1:
        raise ExternalRAOStarError(
            f"External RAO* worker emitted {len(payloads)} result records; expected one."
        )
    try:
        return json.loads(payloads[0])
    except json.JSONDecodeError as error:
        raise ExternalRAOStarError("External RAO* worker emitted invalid JSON.") from error


def run_external_raostar(
    *,
    checkout: Path,
    python: Path,
    seed: int = 0,
    chance_constraint: float | None = None,
    horizon: int | None = None,
    time_limit: float = 300.0,
    iter_limit: int | None = None,
    manifest_path: Path = DEFAULT_MANIFEST,
    process_timeout: float | None = None,
    accept_no_license: bool = False,
) -> dict[str, Any]:
    """Run upstream RAO* in isolation and return its structured result."""
    manifest_file = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_file)
    if (
        manifest["license"].get("status") == "license-not-provided"
        and not accept_no_license
    ):
        raise ExternalRAOStarError(
            "The pinned RAO* repository provides no license. Pass "
            "--accept-no-license only after reviewing the manifest notice."
        )
    provenance = validate_checkout(checkout, manifest)
    # Preserve a virtual-environment launcher symlink.  Resolving it to the
    # base interpreter would silently discard that environment's site-packages.
    interpreter = Path(python).expanduser().absolute()
    if not interpreter.is_file():
        raise ExternalRAOStarError(f"External RAO* Python interpreter does not exist: {interpreter}")
    if time_limit <= 0.0 or not math.isfinite(time_limit):
        raise ExternalRAOStarError("time_limit must be a positive finite number.")
    canonical = manifest["scenarios"]["quad"]["canonical_parameters"]
    chance_constraint = (
        float(canonical["chance_constraint"])
        if chance_constraint is None
        else float(chance_constraint)
    )
    horizon = int(canonical["fixed_horizon"] if horizon is None else horizon)
    if not 0.0 <= chance_constraint <= 1.0:
        raise ExternalRAOStarError("chance_constraint must be in [0, 1].")
    if horizon <= 0:
        raise ExternalRAOStarError("horizon must be positive.")
    if iter_limit is not None and iter_limit <= 0:
        raise ExternalRAOStarError("iter_limit must be positive when provided.")
    try:
        manifest_bytes = manifest_file.read_bytes()
        if json.loads(manifest_bytes) != manifest:
            raise ExternalRAOStarError(
                "RAO* manifest changed while preparing the frozen run inputs."
            )
        worker_bytes = WORKER.read_bytes()
    except (OSError, json.JSONDecodeError) as error:
        raise ExternalRAOStarError(f"Cannot snapshot local RAO* inputs: {error}") from error
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    worker_sha256 = hashlib.sha256(worker_bytes).hexdigest()

    command = [
        str(interpreter),
        "__FROZEN_WORKER__",
        "--manifest",
        "__FROZEN_MANIFEST__",
        "--checkout",
        "__PINNED_SNAPSHOT__",
        "--chance-constraint",
        repr(chance_constraint),
        "--horizon",
        str(horizon),
        "--time-limit",
        repr(float(time_limit)),
    ]
    if iter_limit is not None:
        command.extend(["--iter-limit", str(iter_limit)])
    timeout = process_timeout if process_timeout is not None else time_limit + 30.0
    if timeout <= 0.0 or not math.isfinite(timeout):
        raise ExternalRAOStarError("process_timeout must be a positive finite number.")

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONHASHSEED"] = str(seed)
    environment["OMP_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with (
        tempfile.TemporaryDirectory(prefix="darp-raostar-mpl-") as mpl_config,
        tempfile.TemporaryDirectory(prefix="darp-raostar-source-") as source_snapshot,
        tempfile.TemporaryDirectory(prefix="darp-raostar-local-") as local_snapshot,
    ):
        environment["MPLCONFIGDIR"] = mpl_config
        snapshot = Path(source_snapshot).resolve()
        _archive_pinned_checkout(
            Path(provenance.checkout), provenance.expected_commit, snapshot
        )
        local = Path(local_snapshot).resolve()
        frozen_worker = local / "external_raostar_worker.py"
        frozen_manifest = local / "manifest.json"
        frozen_worker.write_bytes(worker_bytes)
        frozen_manifest.write_bytes(manifest_bytes)
        command[command.index("__FROZEN_WORKER__")] = str(frozen_worker)
        command[command.index("__FROZEN_MANIFEST__")] = str(frozen_manifest)
        command[command.index("__PINNED_SNAPSHOT__")] = str(snapshot)
        try:
            completed = subprocess.run(
                command,
                cwd=str(snapshot),
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise ExternalRAOStarError(
                f"External RAO* process exceeded the hard timeout of {timeout:g}s."
            ) from error
    result = _extract_worker_result(completed.stdout)
    if completed.returncode != 0 and result.get("status") == "ok":
        raise ExternalRAOStarError(
            "External RAO* worker reported status=ok but exited non-zero; "
            "discarding the untrusted result."
        )
    result["provenance"] = provenance.to_dict()
    result["manifest"] = {
        "sha256": manifest_sha256,
    }
    result["local_source_snapshot"] = {
        "worker_sha256": worker_sha256,
    }
    return result
