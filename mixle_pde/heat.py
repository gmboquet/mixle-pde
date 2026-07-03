"""Transient heterogeneous heat conduction for active thermography and conduction inverse problems.

The forward for flash / lock-in thermography and any transient-conduction inversion:

    rho c dT/dt = div(k(x) grad T) + q(x, t)

on a structured grid, marched explicitly in time and differentiable in the conductivity field ``k(x)``,
the volumetric heat capacity ``rho c``, and any driving source ``q``. The spatial operator is the
heterogeneous ``-div(k grad)`` assembled by :func:`mixle_pde.pde_solve.divergence_form` (arithmetic-mean
face conductance), so a subsurface defect (a low-conductivity delamination, a void, an inclusion) enters
as a local contrast in ``k`` and the gradient of a surface-temperature misfit w.r.t. ``k`` flows straight
back through the stepper.

Two boundary geometries cover the thermography setups:

* Dirichlet edges hold a fixed temperature (the default; homogeneous unless ``dirichlet`` is given).
* A surface heat flux ``Q(t)`` deposited on one face -- a flash pulse or a periodic lock-in excitation --
  entering through a half-cell finite-volume balance at the surface nodes. The surface temperature there
  is the observable (an IR camera), read with :meth:`surface`.

The time loop is the checkpointed adjoint integrator :func:`mixle_pde.pde_solve.integrate_record`, so a
long thermogram over a large field stays memory-feasible for the inverse problem. The inverse recovers a
diffusivity / defect contrast from the recorded surface temperature; fit the resulting ``Differential``
with ``how='gauss_newton'``.

The explicit scheme is stable under the conduction CFL limit ``dt <= h^2 / (2 d alpha_max)`` (``d`` the
dimension, ``alpha = k / rho c`` the diffusivity); :meth:`stable_dt` returns a safe step.
"""

from __future__ import annotations

import numpy as np

from mixle_pde.pde_solve import _grid_faces, divergence_form

__all__ = ["TransientHeat"]


class TransientHeat:
    """A differentiable transient heterogeneous heat-conduction stepper (explicit, finite volume).

    ``TransientHeat(n, dt=..., spacing=..., flux_face=...)`` builds the forward on a grid of shape ``n``
    (an int for 1-D or a tuple for n-D). ``step(T, k, ops, rho_c=..., q=..., flux=...)`` advances the
    temperature field one step under ``rho c dT/dt = div(k grad T) + q`` given the per-node conductivity
    field ``k`` (a driver or a fixed array). ``rho c`` is the volumetric heat capacity (scalar or field);
    ``q`` an optional per-node volumetric source for that step; ``flux`` the surface heat flux for that step
    when ``flux_face`` is set.

    Args:
        n: grid shape -- an int ``nx`` (1-D) or a tuple ``(nx, ny, ...)``.
        dt: time step.
        spacing: grid spacing (scalar or per-axis); defaults to ``1/(nx-1)`` for a unit 1-D domain.
        flux_face: ``(axis, side)`` selecting the surface that carries the heat flux, ``side`` in
            ``{0, 1}`` (the low / high face along ``axis``); ``None`` for all-Dirichlet.
    """

    def __init__(self, n, *, dt: float, spacing=None, flux_face: tuple[int, int] | None = None):
        self.shape = (int(n),) if np.isscalar(n) else tuple(int(s) for s in n)
        self.ndim = len(self.shape)
        self.dt = float(dt)
        if spacing is None:
            spacing = 1.0 / (self.shape[0] - 1) if self.ndim == 1 else 1.0
        self.spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (self.ndim,)).copy()
        self.n = int(np.prod(self.shape))
        g = _grid_faces(self.shape, self.spacing)
        self._interior = np.asarray(g["interior"])
        self._boundary = np.asarray(g["boundary"])
        self._boundary_mask = np.asarray(g["boundary_mask"])
        self.flux_face = flux_face
        self._flux_nodes, self._flux_h, self._active = self._resolve_flux(flux_face)

    def _resolve_flux(self, flux_face):
        """Nodes on the flux surface, the spacing normal to it, and the mask of nodes we time-march.

        Flux (Neumann) surface nodes are marched with a half-cell balance rather than pinned, so the set of
        active nodes is the interior plus the flux face; the remaining boundary nodes stay Dirichlet.
        """
        if flux_face is None:
            active = np.zeros(self.n, dtype=bool)
            active[self._interior] = True
            return np.array([], dtype=int), 1.0, active
        axis, side = int(flux_face[0]), int(flux_face[1])
        idx = np.arange(self.n).reshape(self.shape)
        sl = [slice(None)] * self.ndim
        sl[axis] = 0 if side == 0 else self.shape[axis] - 1
        flux_nodes = idx[tuple(sl)].ravel()
        # a flux surface node is a physical corner/edge if it also lies on another boundary axis; keep only
        # the ones interior in every other axis so the half-cell 1-D normal balance is exact.
        keep = np.ones(len(flux_nodes), dtype=bool)
        coords = np.array(np.unravel_index(flux_nodes, self.shape)).T
        for other in range(self.ndim):
            if other == axis:
                continue
            keep &= (coords[:, other] > 0) & (coords[:, other] < self.shape[other] - 1)
        flux_nodes = flux_nodes[keep]
        active = np.zeros(self.n, dtype=bool)
        active[self._interior] = True
        active[flux_nodes] = True
        return flux_nodes, float(self.spacing[axis]), active

    def stable_dt(self, alpha_max: float, *, safety: float = 0.9) -> float:
        """The explicit conduction CFL step ``safety * h^2 / (2 d alpha_max)`` for max diffusivity ``alpha_max``."""
        hmin = float(np.min(self.spacing))
        return safety * hmin**2 / (2.0 * self.ndim * float(alpha_max))

    def _rho_c_field(self, rho_c, ops):
        if np.isscalar(rho_c) or (hasattr(rho_c, "ndim") and getattr(rho_c, "ndim", 1) == 0):
            return ops.tensor(np.full(self.n, float(rho_c)))
        return rho_c

    def surface(self, T):
        """The temperatures on the flux surface (the thermography observable). Requires ``flux_face``."""
        if self.flux_face is None:
            raise ValueError("surface() requires a flux_face (the observed thermography surface).")
        import torch

        return T[torch.as_tensor(self._flux_nodes, dtype=torch.long)]

    def step(self, T, k, ops, *, rho_c=1.0, q=0.0, flux=0.0):
        """Advance the temperature field one explicit step.

        ``rho c dT/dt = div(k grad T) + q`` on the interior; on a flux surface the half-cell balance adds
        ``2 Q / h`` (the flux ``Q`` deposited over a cell of width ``h/2``). Dirichlet nodes are held fixed.
        """
        import torch

        rho_c = self._rho_c_field(rho_c, ops)
        rows, cols, vals, n = divergence_form(k, self.shape, spacing=self.spacing)
        # div(k grad T) = -(A T) on the interior (A's boundary rows are the identity, so ignore them there).
        aT = ops.matvec(rows, cols, vals, n, T)
        div_kgradT = -aT

        q_field = q if torch.is_tensor(q) else ops.tensor(np.full(self.n, float(q)))
        rhs = div_kgradT + q_field  # rho c dT/dt on the interior

        if len(self._flux_nodes):
            fn = torch.as_tensor(self._flux_nodes, dtype=torch.long)
            # half-cell surface balance: rho c (h/2) dT/dt = Q - k (T_s - T_in)/h + q (h/2).
            # the conduction term k (T_in - T_s)/h^2 already sits in -(A T) once the surface node is a
            # normal interior-style Laplacian row; divergence_form gave it an identity row, so rebuild it.
            axis = int(self.flux_face[0])
            side = int(self.flux_face[1])
            neigh = self._inward_neighbor(self._flux_nodes, axis, side)
            neigh_t = torch.as_tensor(neigh, dtype=torch.long)
            h = self._flux_h
            k_face = 0.5 * (k[fn] + k[neigh_t])
            cond = k_face * (T[neigh_t] - T[fn]) / h**2  # k (T_in - T_s)/h^2, one-sided (half cell => x2)
            flux_t = flux if torch.is_tensor(flux) else float(flux)
            surface_rhs = 2.0 * cond + 2.0 * self._flux_face_flux(flux_t, fn, ops) / h
            # overwrite the surface-node rhs (its identity row from A gave -(A T) = -T there)
            rhs = rhs.index_copy(0, fn, surface_rhs + q_field[fn])

        dTdt = rhs / rho_c
        active = torch.as_tensor(self._active)
        return torch.where(active, T + self.dt * dTdt, T)

    def _flux_face_flux(self, flux, fn, ops):
        import torch

        if torch.is_tensor(flux):
            return flux if flux.ndim else flux.expand(len(fn))
        return ops.tensor(np.full(len(fn), float(flux)))

    def _inward_neighbor(self, nodes, axis, side):
        """For each flux-surface node the flat index of its inward neighbour along ``axis``."""
        coords = np.array(np.unravel_index(nodes, self.shape))
        coords[axis] += 1 if side == 0 else -1
        return np.ravel_multi_index(coords, self.shape)
