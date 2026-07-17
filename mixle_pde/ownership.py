"""Exhaustive migration dispositions for the integrated PDE compatibility profile."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.resources import files


@dataclass(frozen=True)
class ModuleDisposition:
    module: str
    disposition: str
    final_owner: str
    rationale: str

    def __post_init__(self) -> None:
        if self.disposition not in {"preserve", "reference", "adapt", "migrate", "deprecate", "retire"}:
            raise ValueError("unsupported module disposition")
        if not self.module.startswith("mixle_pde.") or not self.final_owner.startswith("PRJ-") or not self.rationale:
            raise ValueError("module disposition requires module, project owner, and rationale")


_SIM_MODULES = {
    "boundaries",
    "fem",
    "geometry_to_mesh",
    "mesh",
    "multiphysics",
    "shape",
    "simulation_service",
}
_CORE_MODULES = {
    "adjoint",
    "blocky_priors",
    "field_assimilation",
    "field_gauss_newton",
    "field_inversion",
    "field_mcmc",
    "field_priors",
    "inverse",
    "latent",
    "linear_solve",
    "misfit",
    "model_error",
    "observations",
    "posterior_calibration",
    "posterior_query",
    "sample_update",
    "surrogate",
    "uq_lowrank",
}
_DATA_MODULES = {"env_data", "geo_observations"}
_INQUIRY_MODULES = {
    "decision_quantities",
    "informativeness",
    "model_selection",
    "monitoring_design",
    "reasoning",
    "voi",
}
_PROFILE_MODULES = {"_operator", "capabilities", "canonical_adapter", "ops", "ownership", "problem_adapter", "tools"}
_REFERENCE_PROFILE_MODULES = {"multiphysics_reference"}
_MLOPS_MODULES = {"reference_lifecycle"}


def _classify(name: str) -> ModuleDisposition:
    module = f"mixle_pde.{name}"
    if name.endswith("_test") or (len(name) > 2 and name[0] == "c" and name[1].isdigit() and name.endswith("test")):
        return ModuleDisposition(
            module,
            "retire",
            "PRJ-PDE",
            "Package-internal test surface moves to the external test corpus after an evidence window.",
        )
    if name in _PROFILE_MODULES:
        return ModuleDisposition(
            module,
            "preserve",
            "PRJ-PDE",
            "This module is part of the integrated compatibility, capability, or backend profile.",
        )
    if name in _REFERENCE_PROFILE_MODULES:
        return ModuleDisposition(
            module,
            "reference",
            "PRJ-PDE",
            "This narrowly declared executable reference backend remains a curated PDE specialist profile.",
        )
    if name in _MLOPS_MODULES:
        return ModuleDisposition(
            module,
            "migrate",
            "PRJ-MLOPS",
            "The local lifecycle proves bounded control semantics; production job orchestration belongs to MLOps.",
        )
    if name in _SIM_MODULES:
        return ModuleDisposition(
            module,
            "adapt",
            "PRJ-SIM",
            "Canonical numerical representation belongs to Sim; the PDE import remains a tested compatibility/reference adapter.",
        )
    if name in _CORE_MODULES:
        return ModuleDisposition(
            module,
            "migrate",
            "PRJ-CORE",
            "Generic inference, value, posterior, or uncertainty semantics belong to Core; PDE retains compatibility until migration gates pass.",
        )
    if name in _DATA_MODULES:
        return ModuleDisposition(
            module,
            "migrate",
            "PRJ-DATA",
            "Acquisition and observation-data fulfillment belong to Data; solver-facing adapters remain here temporarily.",
        )
    if name in _INQUIRY_MODULES:
        return ModuleDisposition(
            module,
            "migrate",
            "PRJ-INQUIRY",
            "Question decomposition and decision reasoning belong to Inquiry; numerical kernels remain independently callable.",
        )
    return ModuleDisposition(
        module,
        "reference",
        "PRJ-PDE",
        "Validated specialist numerical kernel remains in the curated PDE backend/profile.",
    )


def migration_inventory() -> tuple[ModuleDisposition, ...]:
    """Classify every shipped source module without importing it."""

    names = sorted(
        item.name[:-3]
        for item in files("mixle_pde").iterdir()
        if item.name.endswith(".py") and item.name != "__init__.py"
    )
    result = tuple(_classify(name) for name in names)
    if len(result) != len({item.module for item in result}):
        raise RuntimeError("migration inventory produced duplicate module records")
    return result


def migration_inventory_digest() -> str:
    payload = [asdict(item) for item in migration_inventory()]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["ModuleDisposition", "migration_inventory", "migration_inventory_digest"]
