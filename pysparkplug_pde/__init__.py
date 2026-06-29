"""pysparkplug-pde — PDE/ODE-constrained Bayesian inverse problems, a mixle.ppl plugin.

Importing this package wires the PDE stack into pysparkplug's PPL surface through mixle's extension
hooks (it does not patch mixle): importing :mod:`pysparkplug_pde.pde` fires its
``register_composite("PDEStateSpace", ..., fit_fn=pde_fit)`` so ``PDE(operator).fit(data)`` works, and
the dynamics operators register through ``register_dynamics_operator``. mixle core stays PDE-free; this
package depends on mixle, never the reverse.

    import pysparkplug_pde as pde
    from pysparkplug_pde import PDE, DiffusionOperator
    model = PDE(DiffusionOperator(0.1, n)).fit(field, dt=0.1)

The forward-model infrastructure (``_operator``, ``ops``, ``dynamics``, ``pde_solve``, ``inverse``,
``multiphysics``, ``pde``) and the concrete equations (``wave``/``wave_pml``, ``flow``/``spectral_flow``,
``gas_dynamics``, ``schrodinger``, ``fem``, ``shape``) are the modules; the headline solvers are
re-exported here.
"""

from __future__ import annotations

from typing import Any

from mixle.ppl.core import RandomVariable

# Importing `pde` fires register_composite("PDEStateSpace", ..., fit_fn=pde_fit) into mixle's registry.
from pysparkplug_pde import pde  # noqa: F401
from pysparkplug_pde.dynamics import (
    AdvectionDiffusionOperator,
    AdvectionOperator,
    DiffusionOperator,
    DynamicsOperator,
    available_dynamics_operators,
    make_operator,
    register_dynamics_operator,
)
from pysparkplug_pde.flow import NavierStokes2D
from pysparkplug_pde.geophysics import (
    gravity_point_sensitivity,
    magnetic_dipole_sensitivity,
    depth_weighting,
    cross_gradient,
    dc_resistivity,
    joint_inversion,
    regularized_gauss_newton,
    roughness_operator,
    straight_ray_operator,
)
from pysparkplug_pde.inverse import Differential
from pysparkplug_pde.multiphysics import CoupledPDESystem, solve_poisson
from pysparkplug_pde.shape import level_set_material, shape_optimize
from pysparkplug_pde.wave import WaveEquation2D

# Register the sparse-solve detector so mixle.ppl.field can guard how='laplace' on a sparse PDE forward
# (its dense double-backward Hessian would be silently wrong). mixle has no PDE dependency; this plugs in.
from mixle.ppl.field import register_sparse_solve_detector as _register_sparse_solve_detector
from pysparkplug_pde.pde_solve import sparse_used_since as _sparse_used_since

_register_sparse_solve_detector(_sparse_used_since)


def PDE(operator: Any, *, name: str | None = None) -> RandomVariable:
    """PDE-constrained latent-field model for spatiotemporal data.

    ``operator`` is a :class:`~pysparkplug_pde.dynamics.DynamicsOperator` (e.g. ``DiffusionOperator``,
    ``AdvectionOperator``) whose method-of-lines discretization fixes the linear state transition. Fit
    on a ``(T, m)`` array of noisy field observations: the Kalman/RTS smoother recovers the latent field
    and EM estimates the process/observation noise while the physics-derived dynamics are held fixed.
    Lowers to the ``PDEStateSpace`` family registered (with its fit_fn) when this package is imported.
    """
    return RandomVariable._sample("PDEStateSpace", (operator,), name=name)


__all__ = [
    "PDE",
    "DiffusionOperator",
    "AdvectionOperator",
    "AdvectionDiffusionOperator",
    "DynamicsOperator",
    "make_operator",
    "register_dynamics_operator",
    "available_dynamics_operators",
    "NavierStokes2D",
    "WaveEquation2D",
    "Differential",
    "CoupledPDESystem",
    "solve_poisson",
    "shape_optimize",
    "level_set_material",
]
