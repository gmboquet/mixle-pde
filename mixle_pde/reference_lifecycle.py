"""Durable bounded lifecycle operations for the multiphysics reference study."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mixle_pde.multiphysics_reference import (
    CancellationFailure,
    EvidenceFailure,
    ObservationValue,
    ReferenceData,
    ReferenceExecutionError,
    ResourceFailure,
    build_validated_surrogate,
    canonical_digest,
    manufactured_solution_refinement,
    prepare_reference_study,
    run_bayesian_inversion,
    solve_reference_fem,
    solve_reference_forward,
)


@dataclass(frozen=True)
class ResourceBudget:
    maximum_grid_points: int = 401
    maximum_coupling_iterations: int = 100
    maximum_cells_per_region: int = 32

    def __post_init__(self) -> None:
        if min(self.maximum_grid_points, self.maximum_coupling_iterations, self.maximum_cells_per_region) <= 0:
            raise ValueError("resource budget limits must be positive")


class ReferenceStudyStore:
    """A local JSON artifact store with idempotent, restart-safe study operations."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, study_id: str) -> Path:
        if not study_id.startswith("study-") or not study_id[6:].isalnum():
            raise KeyError("invalid study id")
        return self.root / f"{study_id}.json"

    @staticmethod
    def _event(record: dict[str, Any], kind: str, detail: dict[str, Any] | None = None) -> None:
        sequence = len(record["events"]) + 1
        record["events"].append(
            {
                "id": canonical_digest({"study": record["id"], "sequence": sequence, "kind": kind})[:24],
                "sequence": sequence,
                "kind": kind,
                "at": datetime.now(UTC).isoformat(),
                "detail": detail or {},
            }
        )

    def _write(self, record: dict[str, Any]) -> None:
        path = self._path(record["id"])
        descriptor, temporary = tempfile.mkstemp(prefix=f".{record['id']}-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(record, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _read(self, study_id: str) -> dict[str, Any]:
        path = self._path(study_id)
        if not path.exists():
            raise KeyError(study_id)
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)

    @staticmethod
    def _data(payload: dict[str, Any]) -> ReferenceData:
        return ReferenceData(
            tuple(ObservationValue(**item) for item in payload["observations"]),
            payload.get("truth_conductivity"),
        )

    def create(
        self,
        problem: dict[str, Any],
        problem_hash: str,
        data: ReferenceData,
        *,
        code_revision: str,
        idempotency_key: str,
        grid_points: int = 381,
        coupling_iterations: int = 80,
        cells_per_region: int = 8,
        budget: ResourceBudget = ResourceBudget(),
    ) -> str:
        if not idempotency_key:
            raise ValueError("idempotency key is required")
        study = prepare_reference_study(problem, problem_hash, data, code_revision=code_revision)
        identity_payload = {
            "problem": problem_hash,
            "data": asdict(data),
            "code_revision": code_revision,
            "idempotency_key": idempotency_key,
        }
        study_id = f"study-{canonical_digest(identity_payload)[:20]}"
        path = self._path(study_id)
        if path.exists():
            existing = self._read(study_id)
            if existing["request_identity"] != canonical_digest(identity_payload):
                raise ValueError("study id collision")
            return study_id
        record = {
            "schema_version": "1.0.0",
            "id": study_id,
            "request_identity": canonical_digest(identity_payload),
            "status": "created",
            "phase": "created",
            "attempt": 1,
            "problem": problem,
            "problem_hash": problem_hash,
            "data": asdict(data),
            "code_revision": code_revision,
            "mode": study.mode.value,
            "identities": asdict(study.identities),
            "configuration": {
                "grid_points": grid_points,
                "coupling_iterations": coupling_iterations,
                "cells_per_region": cells_per_region,
            },
            "budget": asdict(budget),
            "checkpoint": None,
            "result": None,
            "failure": None,
            "events": [],
        }
        self._event(record, "created", {"mode": study.mode.value})
        self._write(record)
        return study_id

    def inspect(self, study_id: str) -> dict[str, Any]:
        return self._read(study_id)

    def monitor(self, study_id: str) -> dict[str, Any]:
        record = self._read(study_id)
        return {
            "id": record["id"],
            "status": record["status"],
            "phase": record["phase"],
            "attempt": record["attempt"],
            "last_event": record["events"][-1],
            "failure": record["failure"],
        }

    def cancel(self, study_id: str) -> dict[str, Any]:
        record = self._read(study_id)
        if record["status"] == "complete":
            raise CancellationFailure("E_ALREADY_COMPLETE", "a completed study cannot be cancelled")
        if record["status"] != "cancelled":
            record["status"] = "cancelled"
            self._event(record, "cancelled", {"phase": record["phase"]})
            self._write(record)
        return self.monitor(study_id)

    def resume(
        self,
        study_id: str,
        *,
        grid_points: int | None = None,
        coupling_iterations: int | None = None,
        cells_per_region: int | None = None,
    ) -> dict[str, Any]:
        record = self._read(study_id)
        if record["status"] not in {"cancelled", "failed"}:
            raise CancellationFailure("E_NOT_RESUMABLE", "only cancelled or failed studies can be resumed")
        overrides = {
            "grid_points": grid_points,
            "coupling_iterations": coupling_iterations,
            "cells_per_region": cells_per_region,
        }
        for key, value in overrides.items():
            if value is not None:
                record["configuration"][key] = value
        record["attempt"] += 1
        record["status"] = "created"
        record["failure"] = None
        self._event(record, "resumed", {"configuration": record["configuration"]})
        self._write(record)
        return self.monitor(study_id)

    @staticmethod
    def _check_budget(record: dict[str, Any]) -> None:
        requested = record["configuration"]
        budget = record["budget"]
        pairs = (
            ("grid_points", "maximum_grid_points"),
            ("coupling_iterations", "maximum_coupling_iterations"),
            ("cells_per_region", "maximum_cells_per_region"),
        )
        exceeded = {
            requested_name: {"requested": requested[requested_name], "limit": budget[budget_name]}
            for requested_name, budget_name in pairs
            if requested[requested_name] > budget[budget_name]
        }
        if exceeded:
            raise ResourceFailure(
                "E_RESOURCE_BUDGET", "study request exceeds its declared budget", diagnostics=exceeded
            )

    def run(self, study_id: str) -> dict[str, Any]:
        record = self._read(study_id)
        if record["status"] == "cancelled":
            raise CancellationFailure("E_CANCELLED", "resume the cancelled study before running it")
        if record["status"] == "complete":
            return self.monitor(study_id)
        if record["status"] != "created":
            raise CancellationFailure("E_INVALID_STATE", f"study cannot run from state {record['status']!r}")
        record["status"] = "running"
        self._event(record, "running")
        self._write(record)
        try:
            self._check_budget(record)
            data = self._data(record["data"])
            study = prepare_reference_study(
                record["problem"],
                record["problem_hash"],
                data,
                code_revision=record["code_revision"],
            )
            config = record["configuration"]
            truth = data.truth_conductivity if data.truth_conductivity is not None else 0.5
            forward = solve_reference_forward(
                study,
                truth,
                maximum_iterations=config["coupling_iterations"],
            )
            fem = solve_reference_fem(study, truth, cells_per_region=config["cells_per_region"])
            refinement = manufactured_solution_refinement()
            record["phase"] = "forward-verified"
            record["checkpoint"] = {
                "phase": record["phase"],
                "forward_identity": forward.solution_identity,
                "fem_identity": fem.forward.solution_identity,
            }
            self._event(record, "checkpoint", record["checkpoint"])
            self._write(record)
            surrogate = build_validated_surrogate(study)
            posterior = run_bayesian_inversion(study, grid_points=config["grid_points"], surrogate=surrogate)
            if data.truth_conductivity is not None and not posterior.truth_recovered:
                raise EvidenceFailure(
                    "E_TRUTH_RECOVERY",
                    "synthetic truth is outside the preregistered posterior credible interval",
                    diagnostics={"interval": posterior.credible_interval, "truth": data.truth_conductivity},
                )
            coupling_history = tuple(item.flux_residual_W_per_m2 for item in forward.coupling_history) or (0.0,)
            result = {
                "identities": asdict(study.identities),
                "compatibility": asdict(study.compatibility),
                "forward": asdict(forward),
                "fem": asdict(fem),
                "refinement": asdict(refinement),
                "surrogate": asdict(surrogate),
                "posterior": asdict(posterior),
                "discrete_evidence": {
                    "residual_vector": list(fem.forward.residual_vector),
                    "oriented_interface_fluxes": list(fem.forward.oriented_interface_fluxes),
                    "interface_values": list(fem.forward.interface_temperatures_K),
                    "refinement_scales": list(refinement.scales),
                    "refinement_errors": list(refinement.l2_errors),
                    "coupling_residual_history": list(coupling_history),
                    "posterior": [
                        {
                            "value": item.value,
                            "weight": item.weight,
                            "provenance_digest": item.provenance_digest,
                        }
                        for item in posterior.samples
                    ],
                    "truth_value": data.truth_conductivity,
                    "prior_bounds": list(study.conductivity_bounds),
                    "identifiability_warning": posterior.identifiability_warning,
                    "diagnostics": {
                        "posterior_summary_identity": posterior.summary_identity,
                        "forward_solution_identity": forward.solution_identity,
                    },
                },
            }
            record["status"] = "complete"
            record["phase"] = "complete"
            record["checkpoint"] = {
                "phase": "complete",
                "posterior_summary_identity": posterior.summary_identity,
            }
            record["result"] = result
            self._event(record, "complete", record["checkpoint"])
            self._write(record)
        except ReferenceExecutionError as error:
            record = self._read(study_id)
            record["status"] = "failed"
            record["failure"] = {
                "category": error.category,
                "code": error.code,
                "message": str(error),
                "diagnostics": error.diagnostics,
            }
            self._event(record, "failed", record["failure"])
            self._write(record)
        return self.monitor(study_id)

    def retrieve(self, study_id: str) -> dict[str, Any]:
        record = self._read(study_id)
        if record["status"] != "complete" or record["result"] is None:
            raise EvidenceFailure(
                "E_RESULT_UNAVAILABLE",
                "result is available only after successful completion",
                diagnostics={"status": record["status"], "phase": record["phase"]},
            )
        return record["result"]


__all__ = ["ReferenceStudyStore", "ResourceBudget"]
