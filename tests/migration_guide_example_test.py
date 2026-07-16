"""Executes the worked example from docs/migrations/legacy-to-canonical-adapters.md.

The migration guide claims that driving mixle_pde.wave.WaveEquation2D through its old direct API and
through the new mixle_pde.pde_backend_registry canonical adapter path, with identical parameters, produces
numerically identical output -- and that a study asking for evidence the backend does not declare is
rejected rather than silently downgraded. This test runs both snippets from the guide verbatim (mirrored
here, not imported from the doc) so that claim is verified on every test run, not just asserted in prose.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle_pde.pde_backend_registry import run_math_problem
from mixle_pde.problem_adapter import UnsupportedPDEProblem

torch = pytest.importorskip("torch", reason="the wave kernel uses the differentiable ops backend")


_N = 16
_DT = 0.02
_N_STEPS = 10
_WAVE_SPEED = 1.0
_AMPLITUDE = 1.0
_SOURCE_NODE = (_N // 2) * _N + _N // 2


def _legacy_direct_displacement() -> np.ndarray:
    """The "old direct API" snippet from the migration guide, unmodified."""
    from mixle_pde.ops import make_ops
    from mixle_pde.wave import WaveEquation2D

    wave = WaveEquation2D(_N, dt=_DT)
    ops = make_ops()
    state = wave.pack(
        torch.zeros(_N * _N, dtype=torch.float64),
        torch.zeros(_N * _N, dtype=torch.float64),
    )
    c2 = _WAVE_SPEED**2
    for step in range(_N_STEPS):
        source = ops.zeros(_N * _N)
        if step == 0:
            source = source.clone()
            source[_SOURCE_NODE] = _AMPLITUDE
        state = wave.step(state, c2, ops, source=source)

    return wave.displacement(state).detach().numpy()


def _wave_migration_problem(*, evidence_kinds: tuple[str, ...] = ("convergence",)) -> dict:
    """The "new canonical adapter path" study dictionary from the migration guide, unmodified."""
    return {
        "id": "wave-migration-demo",
        "domains": [{"id": "domain", "kind": "mesh", "properties": {"mesh_cell_type": "structured_grid"}}],
        "unknowns": [{"id": "field", "domain_id": "domain"}],
        "operators": [
            {
                "id": "wave-migration-demo-operator",
                "kind": "time_stepping",
                "input_ids": ["field"],
                "output_ids": ["field"],
                "discretization": "FD-leapfrog",
            }
        ],
        "constraints": [],
        "objectives": [{"id": "solution", "sense": "satisfy", "expression": {}}],
        "evidence_requests": [{"kind": kind, "required": True} for kind in evidence_kinds],
        "solve_plan": {
            "parameters": {
                "grid_size": _N,
                "dt": _DT,
                "n_steps": _N_STEPS,
                "wave_speed": _WAVE_SPEED,
                "amplitude": _AMPLITUDE,
                "source_node": _SOURCE_NODE,
            }
        },
    }


def test_legacy_direct_and_canonical_adapter_paths_agree_exactly():
    legacy_displacement = _legacy_direct_displacement()

    result = run_math_problem(_wave_migration_problem(), "wave-fd-leapfrog")

    assert result.compatibility_report.supported
    assert result.evidence["convergence"]["finite"] is True
    assert isinstance(result.solution, np.ndarray)
    assert result.solution.shape == legacy_displacement.shape
    # Both paths drive the identical WaveEquation2D stepper with identical parameters: the guide's claim is
    # exact numerical agreement, not merely approximate agreement.
    assert np.max(np.abs(legacy_displacement - result.solution)) == 0.0


def test_canonical_adapter_path_rejects_evidence_the_backend_does_not_declare():
    # wave-fd-leapfrog only declares "convergence" evidence; asking for "residual" must be rejected rather
    # than silently solved and answered with whatever evidence happens to be available.
    problem = _wave_migration_problem(evidence_kinds=("residual",))

    with pytest.raises(UnsupportedPDEProblem) as excinfo:
        run_math_problem(problem, "wave-fd-leapfrog")

    assert excinfo.value.report.unsupported_features == ("evidence:residual",)
