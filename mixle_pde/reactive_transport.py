"""Tailings seepage + acid-mine-drainage (AMD) reactive transport (workstream G5).

Sulfide-bearing tailings (pyrite/chalcopyrite waste rock) oxidize on contact with water and
oxygen, releasing sulfate, mobilizing metals (Fe, Cu, ...), and generating acidity -- the classic
acid-mine-drainage reaction, approximately

    FeS2 + 15/4 O2 + 7/2 H2O -> Fe(OH)3 + 2 SO4^2- + 4 H+

:func:`amd_reaction` is a first-cut kinetic model for that process: pseudo-first-order sulfide
consumption with stoichiometric release of sulfate, metal, and protons. :class:`ReactiveTransport`
couples it to a G1-style transport operator (any :class:`mixle_pde.dynamics.DynamicsOperator` --
:class:`mixle_pde.groundwater.GroundwaterTransportOperator` once available, or its
:class:`~mixle_pde.dynamics.AdvectionDiffusionOperator` base in the meantime) by Strang
operator-splitting: half a step of advection-dispersion, a full (implicit, Newton-solved) reaction
step, then the second transport half-step. A tailings/dam seepage boundary condition injects the
reactive source at the column inlet.

This is deliberately a "first-cut" screening model, not a geochemical-equilibrium database: see the
module docstring in ``notes/exec/workstream-G.md`` (G5 Non-goals) for the explicit scope limits
(no PHREEQC-parity thermodynamics, no multiphase/unsaturated flow, no microbial kinetics).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from mixle_pde.geo_observations import MultiElementAssay, additive_log_ratio, multi_element_assay_log_likelihood
from mixle_pde.nonlinear import nonlinear_solve

__all__ = [
    "AMD_SPECIES",
    "amd_reaction",
    "ph_from_concentration",
    "amd_reactions_step",
    "ReactiveTransport",
    "effluent_assay",
    "effluent_log_likelihood",
]

# Fixed per-cell species layout every state array in this module uses: remaining oxidizable
# sulfide substrate, sulfate produced, mobilized metal (Fe or Cu -- whichever the caller's
# ``stoichiometry`` names), and proton (H+) concentration.
AMD_SPECIES = ("sulfide", "SO4", "metal", "H")

# Empirical acid-catalysis constants for the optional ``ph`` acceleration term: oxidation kinetics
# accelerate as pH drops below neutral (the Fe(III)-catalyzed pathway dominates in acidic water),
# and are not decelerated above it (accel is clipped to >= 1).
_ACID_CATALYSIS_EXPONENT = 0.5
_ACID_CATALYSIS_REFERENCE_PH = 7.0

_METAL_KEYS = ("Fe", "Cu")


def _metal_stoichiometry(stoichiometry: Mapping[str, float]) -> float:
    for key in _METAL_KEYS:
        if key in stoichiometry:
            return float(stoichiometry[key])
    return 0.0


def ph_from_concentration(h_concentration: np.ndarray | float, *, eps: float = 1e-300) -> np.ndarray:
    """``pH = -log10([H+])`` for a proton concentration in mol/L (elementwise)."""
    h = np.clip(np.asarray(h_concentration, dtype=float), eps, None)
    return -np.log10(h)


def _acid_catalysis(ph: np.ndarray | float | None) -> np.ndarray | float:
    if ph is None:
        return 1.0
    ph_arr = np.asarray(ph, dtype=float)
    return np.power(10.0, np.clip(_ACID_CATALYSIS_EXPONENT * (_ACID_CATALYSIS_REFERENCE_PH - ph_arr), 0.0, None))


def amd_reaction(
    concentrations: np.ndarray,
    *,
    rate_const: float,
    stoichiometry: Mapping[str, float],
    ph: np.ndarray | float | None = None,
) -> np.ndarray:
    """Sulfide-oxidation kinetics: rate of change of ``(sulfide, SO4, metal, H)`` concentrations.

    ``concentrations`` is ``(..., 4)`` ordered as :data:`AMD_SPECIES`. The reaction is pseudo
    first-order in the remaining sulfide substrate, ``oxidation_rate = rate_const * sulfide``
    (optionally accelerated at low pH, see below); each product species scales off that one rate
    by its yield in ``stoichiometry`` (mol product per mol sulfide oxidized), e.g.
    ``{"SO4": 2.0, "Fe": 1.0, "H": 4.0}`` for the classical pyrite-oxidation stoichiometry above.
    The metal column is keyed by whichever of ``"Fe"``/``"Cu"`` is present in ``stoichiometry``
    (zero if neither is given).

    ``ph`` is an optional externally supplied ambient pH (scalar or an array broadcastable against
    ``concentrations[..., 0]``) used only to scale the oxidation rate via a simple acid-catalysis
    factor (``10 ** (0.5 * (7 - ph))``, clipped to never slow the reaction down); it is a prescribed
    forcing, not fed back from the state's own H column, so the per-cell kinetics stay linear in
    ``concentrations`` for the (stiff) implicit reaction step in :func:`amd_reactions_step`. Pass
    ``ph=None`` (the default) for the base kinetics with no acid-catalysis acceleration; recover a
    pH profile from the tracked H column with :func:`ph_from_concentration`.

    Returns an array the same shape as ``concentrations`` with ``d[species]/dt``.
    """
    c = np.asarray(concentrations, dtype=float)
    sulfide = np.clip(c[..., 0], 0.0, None)
    accel = _acid_catalysis(ph)
    oxidation_rate = float(rate_const) * accel * sulfide

    d_sulfide = -oxidation_rate
    d_so4 = float(stoichiometry.get("SO4", 0.0)) * oxidation_rate
    d_metal = _metal_stoichiometry(stoichiometry) * oxidation_rate
    d_h = float(stoichiometry.get("H", 0.0)) * oxidation_rate
    return np.stack([d_sulfide, d_so4, d_metal, d_h], axis=-1)


def _reaction_rate_matrix(*, rate_const: float, stoichiometry: Mapping[str, float], ph) -> np.ndarray:
    """The constant ``4x4`` linear map ``R`` such that ``amd_reaction(u; ...) = R @ u`` (per cell).

    ``amd_reaction`` is linear in ``concentrations`` whenever ``ph`` is not derived from the state
    itself (see its docstring), so the reaction ODE ``du/dt = R u`` has a constant Jacobian -- exact
    for the implicit-Euler Newton step below regardless of the initial iterate.
    """
    accel = float(np.asarray(_acid_catalysis(ph)))
    k = float(rate_const) * accel
    r = np.zeros((4, 4), dtype=float)
    r[0, 0] = -k
    r[1, 0] = float(stoichiometry.get("SO4", 0.0)) * k
    r[2, 0] = _metal_stoichiometry(stoichiometry) * k
    r[3, 0] = float(stoichiometry.get("H", 0.0)) * k
    return r


def amd_reactions_step(
    *,
    rate_const: float,
    stoichiometry: Mapping[str, float],
    ph: np.ndarray | float | None = None,
    max_its: int = 50,
    tol: float = 1e-10,
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Build a ``reactions(state, dt) -> new_state`` callable for :class:`ReactiveTransport`.

    Implements the "solve the (stiff) per-cell reaction ODE with ``nonlinear_solve``/Newton"
    algorithm step: one backward-Euler step of :func:`amd_reaction`,
    ``(u_new - u_old)/dt - amd_reaction(u_new) = 0``, solved per cell with
    :func:`mixle_pde.nonlinear.nonlinear_solve`. The per-cell system is block-diagonal (reaction has
    no spatial coupling -- that is handled by the transport half-steps), so cells never interact
    inside this solve; because :func:`amd_reaction` is linear in the state (see its docstring), the
    Newton Jacobian is exact and constant, and the solve converges in a single step regardless of
    ``dt`` or the initial guess -- still routed through the generic Newton machinery so a future
    nonlinear extension (e.g. self-consistent pH feedback, Monod oxygen limitation) drops in without
    changing the call site.
    """
    import torch

    r_np = _reaction_rate_matrix(rate_const=rate_const, stoichiometry=stoichiometry, ph=ph)
    r_t = torch.as_tensor(r_np, dtype=torch.float64)
    eye4 = torch.eye(4, dtype=torch.float64)

    def reactions(state: np.ndarray, dt: float) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        shape = state.shape
        n_cells = int(np.prod(shape[:-1])) if state.ndim > 1 else 1
        u_old = state.reshape(n_cells, 4)
        u_old_t = torch.as_tensor(u_old, dtype=torch.float64)
        theta = u_old_t.reshape(-1).clone()
        dtf = float(dt)

        block = eye4 / dtf - r_t  # dF/du for one cell (constant -- the system is linear)
        block_np = block.numpy()
        rows_blk, cols_blk = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
        rows_list = []
        cols_list = []
        vals_list = []
        for cell in range(n_cells):
            offset = cell * 4
            rows_list.append(rows_blk.ravel() + offset)
            cols_list.append(cols_blk.ravel() + offset)
            vals_list.append(block_np.ravel())
        rows = torch.as_tensor(np.concatenate(rows_list), dtype=torch.long)
        cols = torch.as_tensor(np.concatenate(cols_list), dtype=torch.long)
        base_vals = torch.as_tensor(np.concatenate(vals_list), dtype=torch.float64)

        def residual_fn(u, th):
            u_cells = u.reshape(n_cells, 4)
            old_cells = th.reshape(n_cells, 4)
            rate = u_cells @ r_t.T
            res = (u_cells - old_cells) / dtf - rate
            return res.reshape(-1)

        def jac_fn(u, th):
            # constant Jacobian -- independent of u/theta, block-diagonal per cell.
            return rows, cols, base_vals

        u0 = u_old_t.reshape(-1).clone()
        u_star = nonlinear_solve(residual_fn, jac_fn, u0, theta, max_its=max_its, tol=tol)
        return u_star.detach().cpu().numpy().reshape(shape)

    return reactions


class ReactiveTransport:
    """Couple a G1-style transport operator to AMD reaction kinetics by Strang splitting.

    ``transport`` is any object exposing ``transition_matrix(dt) -> (n, n)`` (the
    :class:`mixle_pde.dynamics.DynamicsOperator` interface -- satisfied by
    :class:`mixle_pde.groundwater.GroundwaterTransportOperator` or, until that lands, its
    :class:`~mixle_pde.dynamics.AdvectionDiffusionOperator` base) advancing an ``(n_cells,)`` or
    ``(n_cells, n_species)`` state one linear advection-dispersion step. ``reactions`` is a
    ``(state, dt) -> new_state`` callable (see :func:`amd_reactions_step`) advancing the same state
    through the local chemistry over a full ``dt``. ``seepage_bc``, if given, applies a tailings/dam
    seepage face boundary condition after each transport half-step: either a callable
    ``seepage_bc(state) -> state``, or a mapping ``{"index": idx, "concentration": value}``
    (prescribed/Dirichlet concentration at ``idx``) or ``{"index": idx, "flux": value}`` (a source
    added as ``flux * dt`` at ``idx``, a Neumann-style influx).

    A concentration-pinning ``seepage_bc`` is applied by overwriting the boundary cell *after* the
    transport half-step, which only reproduces a true Dirichlet condition if ``transport`` advances
    state from old values alone (``scheme="explicit"`` or ``"exact"`` on
    :class:`~mixle_pde.dynamics.DynamicsOperator`). An implicit half-step solves the interior and
    boundary jointly, so it also uses (and is influenced by) whatever the boundary cell's own,
    un-pinned dynamics computed that step -- overwriting the result afterward does not undo that
    influence on the interior. Use ``scheme="explicit"`` (with a stable ``dt`` for the operator's
    Peclet/CFL numbers) when a seepage face needs a hard prescribed concentration.
    """

    def __init__(
        self,
        transport: Any,
        reactions: Callable[[np.ndarray, float], np.ndarray],
        *,
        seepage_bc: Mapping[str, Any] | Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.transport = transport
        self.reactions = reactions
        self.seepage_bc = seepage_bc

    def _transport_half_step(self, state: np.ndarray, half_dt: float) -> np.ndarray:
        a = np.asarray(self.transport.transition_matrix(half_dt), dtype=float)
        return a @ np.asarray(state, dtype=float)

    def _apply_seepage_bc(self, state: np.ndarray, dt: float) -> np.ndarray:
        bc = self.seepage_bc
        if bc is None:
            return state
        if callable(bc):
            return np.asarray(bc(state), dtype=float)
        state = np.array(state, dtype=float, copy=True)
        idx = bc["index"]
        if "concentration" in bc:
            state[idx] = np.asarray(bc["concentration"], dtype=float)
        if "flux" in bc:
            state[idx] = state[idx] + np.asarray(bc["flux"], dtype=float) * float(dt)
        return state

    def step(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Advance ``state`` by ``dt``: half transport, full reaction, half transport (Strang split)."""
        half = 0.5 * float(dt)
        state = self._transport_half_step(state, half)
        state = self._apply_seepage_bc(state, half)
        state = self.reactions(state, float(dt))
        state = self._transport_half_step(state, half)
        state = self._apply_seepage_bc(state, half)
        return np.asarray(state, dtype=float)


def effluent_assay(
    state: np.ndarray,
    sample_index: Any,
    location: np.ndarray,
    *,
    elements: tuple[str, ...] = ("SO4", "metal"),
    noise_cov: np.ndarray | None = None,
    units: str = "",
    provenance: dict[str, Any] | None = None,
    alr: bool = False,
    denom_index: int = -1,
) -> MultiElementAssay:
    """Package sampled effluent cells of a reactive-transport ``state`` as a :class:`MultiElementAssay`.

    ``state`` is ``(n_cells, 4)`` (:data:`AMD_SPECIES` columns); ``sample_index`` selects the
    sampled cells (e.g. the seepage-face outlet) and ``location`` gives their real-world ``(m, 3)``
    coordinates. Only the product columns named in ``elements`` (default sulfate + mobilized metal)
    are reported -- an AMD effluent monitoring panel, not the internal reaction-tracer state. When
    ``alr=True`` the sampled values are additive-log-ratio transformed first (:func:`additive_log_ratio`,
    workstream G3's geochem contract), so a downstream inversion can model them as a composition
    rather than independent concentrations; ``noise_cov`` is then the covariance in ALR space.
    """
    state = np.asarray(state, dtype=float)
    species_index = {name: i for i, name in enumerate(AMD_SPECIES)}
    cols = [species_index[name] for name in elements]
    values = state[sample_index][:, cols]
    values = np.atleast_2d(values)
    if alr:
        values = additive_log_ratio(values, denom_index=denom_index)
        elements = tuple(f"alr_{name}" for i, name in enumerate(elements) if i != (denom_index % len(elements)))
    k = len(elements)
    if noise_cov is None:
        noise_cov = np.full(k, 1e-8)
    return MultiElementAssay(
        elements=elements,
        location=location,
        value=values,
        noise_cov=noise_cov,
        units=units,
        provenance=dict(provenance or {}),
    )


def effluent_log_likelihood(assay: MultiElementAssay, predicted: np.ndarray) -> float:
    """``log p(assay | predicted effluent concentrations)`` -- a thin pass-through to
    :func:`mixle_pde.geo_observations.multi_element_assay_log_likelihood` so a G5 inversion reuses
    the same geochemistry contract as any other assay (workstream G3)."""
    return multi_element_assay_log_likelihood(assay, predicted)
