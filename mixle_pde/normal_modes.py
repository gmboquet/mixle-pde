"""KRAKEN-style depth normal-mode solver for a horizontally-stratified ocean waveguide.

Long-range ocean acoustics is most naturally described by normal modes. In a range-independent waveguide the
pressure separates as ``p(r, z) = sum_m phi_m(z) H_0^{(1)}(k_m r)``, so the field is a sum of trapped depth
modes ``phi_m(z)`` each propagating horizontally with its own wavenumber ``k_m``. The depth modes solve the
Sturm-Liouville eigenproblem

    d^2 phi_m / dz^2 + ( omega^2 / c(z)^2 - k_m^2 ) phi_m = 0,

with a pressure-release surface ``phi(0) = 0`` and a seabed condition at ``z = D`` (rigid, pressure-release,
or a fast fluid halfspace / impedance bottom). The horizontal wavenumbers ``k_m`` are the eigenvalues and
``phi_m(z)`` the mode shapes; only modes with ``k_m`` real (energy trapped in the water column) propagate to
long range, the rest radiate into the bottom and are dropped.

Discretizing the depth with second-order finite differences turns this into a symmetric matrix eigenproblem
``A phi = k^2 phi``, solved differentiably with ``torch.linalg.eigh`` exactly as :mod:`guided_wave` linearizes
the SAFE plate dispersion. The field is then synthesized by the modal sum

    p(r, z) = ( i e^{-i pi/4} / (rho(z_s) sqrt(8 pi r)) ) sum_m phi_m(z_s) phi_m(z) e^{i k_m r} / sqrt(k_m),

the standard far-field (large-``k_m r``) form of the Hankel-function expansion. Because the depth operator is
assembled from ``c(z)`` through the ``ops`` tensors and the eigen-solve is autograd-capable, the whole map
``c(z) -> (k_m, phi_m, transmission loss)`` is differentiable, so a measured field can be inverted for the
sound-speed profile by gradient descent.

Reference: Jensen, Kuperman, Porter & Schmidt, *Computational Ocean Acoustics* (the Pekeris waveguide and the
KRAKEN finite-difference mode solver); Pekeris, "Theory of propagation of explosive sound in shallow water"
(1948), the two-layer characteristic equation the solver is validated against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["NormalModes1D", "ModeSet", "pekeris_characteristic", "pekeris_mode_count"]


@dataclass
class ModeSet:
    """Trapped depth modes of a waveguide at one frequency.

    ``freq`` is the frequency (Hz), ``omega = 2 pi freq``. ``k`` are the real horizontal wavenumbers (rad/m)
    of the trapped modes sorted descending (mode 1 = highest ``k`` = lowest phase speed = grazing-most), ``z``
    the depth axis (m), and ``phi`` the mode shapes with shape ``(n_mode, n_z)``, each normalized to unit
    ``int phi^2 dz``. ``c_ph = omega / k`` are the modal phase speeds (m/s).
    """

    freq: float
    k: np.ndarray
    z: np.ndarray
    phi: np.ndarray

    @property
    def omega(self) -> float:
        return 2.0 * math.pi * float(self.freq)

    @property
    def n_mode(self) -> int:
        return int(self.k.shape[0])

    @property
    def c_ph(self) -> np.ndarray:
        return self.omega / self.k


def pekeris_characteristic(k, omega, D, c1, c2, rho1, rho2):
    """Residual of the Pekeris two-layer characteristic equation; its roots in ``k`` are the trapped modes.

    Isovelocity water (speed ``c1``, density ``rho1``, depth ``D``, pressure-release surface) over a faster
    fluid halfspace (``c2 > c1``, ``rho2``). A trapped mode has ``omega/c2 < k < omega/c1`` so the vertical
    wavenumber in water ``gamma1 = sqrt(omega^2/c1^2 - k^2)`` is real (sinusoidal) and the bottom decay rate
    ``gamma2 = sqrt(k^2 - omega^2/c2^2)`` is real (evanescent). Matching pressure and normal displacement at
    ``z = D`` with ``phi(0) = 0`` gives ``gamma1 cos(gamma1 D) = -(rho1/rho2) gamma2 sin(gamma1 D)``, i.e.

        tan(gamma1 D) = - rho2 gamma1 / (rho1 gamma2).

    Returned as ``gamma1 cos - (rho1/rho2) gamma2 sin`` (finite through the tangent's poles).
    """
    g1 = np.sqrt((omega / c1) ** 2 - k**2 + 0j).real
    g2 = np.sqrt(k**2 - (omega / c2) ** 2 + 0j).real
    return g1 * np.cos(g1 * D) + (rho1 / rho2) * g2 * np.sin(g1 * D)


def pekeris_mode_count(freq, D, c1, c2):
    """Number of trapped modes of a Pekeris waveguide (exact count from the characteristic equation).

    The trapped band is ``omega/c2 < k < omega/c1``; over it ``gamma1 = sqrt(omega^2/c1^2 - k^2)`` sweeps from
    ``gamma1_max = omega sqrt(1/c1^2 - 1/c2^2)`` down to 0. Each half-period of the surface-release mode
    condition contributes one root, giving the exact count

        M = ceil( gamma1_max D / pi ) = ceil( (2 f D / c1) sqrt(1 - (c1/c2)^2) ),

    the familiar ``M ~ (f D / c1) sqrt(1 - (c1/c2)^2) * 2`` estimate made exact.
    """
    omega = 2.0 * math.pi * float(freq)
    g1_max = omega * math.sqrt(1.0 / c1**2 - 1.0 / c2**2)
    return int(math.ceil(g1_max * float(D) / math.pi))


class NormalModes1D:
    """Finite-difference depth normal-mode solver for a range-independent ocean waveguide.

    ``NormalModes1D(depth, c, ...)`` discretizes ``0 <= z <= depth`` with ``n_z`` points and assembles the
    symmetric depth operator ``A`` of the Sturm-Liouville problem ``phi'' + (omega^2/c^2 - k^2) phi = 0``.
    Calling :meth:`solve` at a frequency returns the trapped modes (``k_m`` real, phase speed below the bottom
    speed) via ``torch.linalg.eigh``; :meth:`field` synthesizes the modal-sum pressure and :meth:`transmission_loss`
    the range-depth transmission loss. The sound-speed column flows through the ``ops`` tensors, so the map is
    differentiable in ``c(z)``.

    Args:
        depth: water depth ``D`` (m) of the water column above the seabed.
        c: sound speed ``c(z)`` -- a scalar (isovelocity) or a length-``n_z`` column (m/s); an ``ops`` tensor
            for a differentiable model.
        rho: water density (kg/m^3).
        n_z: number of depth samples (mesh resolution) including both boundaries.
        bottom: seabed condition -- ``"pressure_release"`` (phi(D)=0), ``"rigid"`` (phi'(D)=0), or
            ``"halfspace"`` for a fast fluid halfspace / impedance bottom (needs ``c_bottom``, ``rho_bottom``).
        c_bottom, rho_bottom: halfspace speed (m/s) and density (kg/m^3) for ``bottom="halfspace"``.
        ops: the backend math namespace (``mixle_pde.ops.make_ops()``); created on demand if omitted.
    """

    def __init__(
        self,
        depth: float,
        c: Any,
        *,
        rho: float = 1000.0,
        n_z: int = 400,
        bottom: str = "halfspace",
        c_bottom: float | None = None,
        rho_bottom: float | None = None,
        ops: Any | None = None,
    ):
        if bottom not in ("pressure_release", "rigid", "halfspace"):
            raise ValueError("bottom must be 'pressure_release', 'rigid' or 'halfspace'.")
        if bottom == "halfspace" and (c_bottom is None or rho_bottom is None):
            raise ValueError("bottom='halfspace' needs c_bottom (m/s) and rho_bottom (kg/m^3).")
        self.depth = float(depth)
        self.rho = float(rho)
        self.n_z = int(n_z)
        self.bottom = bottom
        self.c_bottom = None if c_bottom is None else float(c_bottom)
        self.rho_bottom = None if rho_bottom is None else float(rho_bottom)
        if ops is None:
            from mixle_pde.ops import make_ops

            ops = make_ops()
        self.ops = ops
        self.dz = self.depth / (self.n_z - 1)
        self._c = self._as_c_column(c)

    def _torch(self):
        import torch

        return torch

    def depths(self):
        """The depth axis ``[0, dz, ..., depth]`` (m) as a float64 tensor."""
        t = self._torch()
        return t.arange(self.n_z, dtype=t.float64) * self.dz

    def _as_c_column(self, c):
        """Broadcast the sound speed to a length-``n_z`` ops tensor (differentiable if ``c`` already is)."""
        t = self._torch()
        if t.is_tensor(c):
            return c * t.ones(self.n_z, dtype=c.dtype) if c.dim() == 0 else c
        arr = np.asarray(c, dtype=float)
        if arr.ndim == 0:
            return t.full((self.n_z,), float(arr), dtype=t.float64)
        return t.as_tensor(arr, dtype=t.float64)

    # ---- differentiable assembly of the depth operator ----------------------------------------------
    def _n_unknown(self):
        """Number of interior unknowns: surface node is always Dirichlet; the seabed node is an unknown
        unless the bottom is pressure-release (then it too is dropped)."""
        return self.n_z - 2 if self.bottom == "pressure_release" else self.n_z - 1

    def _stiffness_mass(self):
        """Assemble the symmetric stiffness ``S`` (discretizing ``-d^2/dz^2``) and diagonal mass ``M``.

        The generalized eigenproblem is ``(diag(omega^2/c^2) M - S) phi = k^2 M phi``: multiplying the
        Sturm-Liouville equation ``-phi'' = (omega^2/c^2 - k^2) phi`` by the finite-element mass matrix keeps
        the operator symmetric even at the Robin seabed. The surface node is dropped (``phi(0)=0``). The seabed
        node, when it is an unknown, uses a half-cell: stiffness diagonal ``1/dz^2`` and mass weight ``1/2``
        (rigid / halfspace); the Robin flux is added later in :meth:`_generalized_operator`.
        """
        t = self._torch()
        inv_dz2 = 1.0 / self.dz**2
        n = self._n_unknown()
        # interior tridiagonal stiffness for -d^2/dz^2
        main = 2.0 * inv_dz2 * t.ones(n, dtype=t.float64)
        off = -inv_dz2 * t.ones(n - 1, dtype=t.float64)
        mass = t.ones(n, dtype=t.float64)
        if self.bottom != "pressure_release":
            # seabed node N is an unknown with only one interior neighbor (half stencil, half mass)
            main = main.clone()
            main[n - 1] = inv_dz2
            mass = mass.clone()
            mass[n - 1] = 0.5
        S = t.diag(main) + t.diag(off, 1) + t.diag(off, -1)
        return S, mass, n

    def _generalized_operator(self, omega, coeff):
        """The symmetric standard-form operator ``B = Mh^{-1} (diag(omega^2/c^2) M - S) Mh^{-1}`` whose
        eigenvalues are ``k^2``, with ``Mh = sqrt(M)``. ``coeff`` is the Robin seabed flux
        ``phi'(D) = -coeff phi(D)`` (``0`` rigid, ``gamma2`` fast halfspace, unused pressure-release), which
        adds ``coeff/dz`` to the seabed stiffness diagonal (i.e. subtracts it from the operator)."""
        t = self._torch()
        S, mass, n = self._stiffness_mass()
        if self.bottom != "pressure_release" and coeff:
            S = S.clone()
            S[n - 1, n - 1] = S[n - 1, n - 1] + coeff / self.dz
        i0, i1 = (1, self.n_z - 1) if self.bottom == "pressure_release" else (1, self.n_z)
        env = (omega / self._c[i0:i1]) ** 2  # omega^2/c^2 at the unknown nodes
        G = t.diag(env * mass) - S
        inv_mh = 1.0 / t.sqrt(mass)
        B = inv_mh[:, None] * G * inv_mh[None, :]  # symmetric standard eigenproblem B phi = k^2 phi
        return 0.5 * (B + B.T), mass, n

    # ---- eigen-solve --------------------------------------------------------------------------------
    def solve(self, freq: float, *, max_iter: int = 30, tol: float = 1e-8) -> ModeSet:
        """Solve the depth eigenproblem at ``freq`` (Hz) and return the trapped :class:`ModeSet`.

        ``A phi = k^2 phi`` is solved with the symmetric ``torch.linalg.eigh``; a mode is trapped if its
        ``k^2`` is real-positive and (for a penetrable bottom) its phase speed lies below the bottom speed,
        i.e. ``omega/c_bottom < k < omega/c_min``. For the halfspace bottom the operator depends weakly on
        ``k`` through the radiation term, so it is refined by a fixed-point iteration on the mode wavenumbers.
        """
        t = self._torch()
        omega = 2.0 * math.pi * float(freq)
        c_min = float(t.min(self._c).item())
        k_water = omega / c_min  # upper bound on any trapped k

        if self.bottom == "halfspace":
            k_cut = omega / self.c_bottom
            k_self = self._self_consistent_wavenumbers(omega, k_water, k_cut, max_iter, tol)
            # rebuild each mode's eigenvector at its own self-consistent halfspace coefficient
            k_list, phi_cols, mass = self._modes_at_self_consistent(omega, k_self, k_water, k_cut)
            k = t.as_tensor(k_list, dtype=t.float64)
            return self._pack_modes(k, phi_cols, mass, freq)
        k_cut = 0.0
        B, mass, _ = self._generalized_operator(omega, 0.0)
        evals, evecs = t.linalg.eigh(B)
        return self._assemble_modes(evals, evecs, mass, freq, k_water, k_cut)

    def _self_consistent_wavenumbers(self, omega, k_water, k_cut, max_iter, tol):
        """Per-mode fixed point: each trapped ``k_m`` must satisfy ``k_m^2 = eig`` of the operator built with
        that mode's own halfspace coefficient ``gamma2(k_m)``. A sweep of trial coefficients seeds the set,
        then every mode is polished to self-consistency independently (its gamma2 depends on its own k)."""
        t = self._torch()
        # seed from two trial coefficients: a near-cutoff one exposes the weakly-trapped high-order modes,
        # a mid-band one the grazing low-order modes. Their union (deduped) covers every trapped mode.
        seeds: list[float] = []
        for frac in (0.03, 0.55):
            k_trial = k_cut + frac * (k_water - k_cut)
            B, _, _ = self._generalized_operator(omega, self._gamma2(omega, k_trial))
            seeds.extend(self._trapped_wavenumbers(t.linalg.eigvalsh(B), k_water, k_cut).tolist())
        seeds = self._dedupe(sorted(seeds, reverse=True), k_water)
        out: list[float] = []
        for k_seed in seeds:
            k_m = k_seed
            for _ in range(int(max_iter)):
                B, _, _ = self._generalized_operator(omega, self._gamma2(omega, k_m))
                ev = self._trapped_wavenumbers(t.linalg.eigvalsh(B), k_water, k_cut)
                if ev.numel() == 0:
                    break
                k_new = float(ev[t.argmin(t.abs(ev - k_m))].item())
                converged = abs(k_new - k_m) < tol * k_m
                k_m = k_new
                if converged:
                    break
            if k_cut < k_m < k_water and not any(abs(k_m - kk) < 1e-6 * k_m for kk in out):
                out.append(k_m)
        return sorted(out, reverse=True)

    @staticmethod
    def _dedupe(vals, scale):
        out: list[float] = []
        for v in vals:
            if not any(abs(v - u) < 1e-4 * scale for u in out):
                out.append(v)
        return out

    def _modes_at_self_consistent(self, omega, k_self, k_water, k_cut):
        """Eigenvectors for each self-consistent ``k_m`` (operator rebuilt at that mode's own gamma2)."""
        t = self._torch()
        cols = []
        ks = []
        mass_ref = None
        for k_m in k_self:
            coeff = self._gamma2(omega, k_m)
            B, mass, _ = self._generalized_operator(omega, coeff)
            mass_ref = mass
            evals, evecs = t.linalg.eigh(B)
            k = t.sqrt(t.clamp(evals, min=0.0))
            j = int(t.argmin(t.abs(k - k_m)).item())
            ks.append(float(k[j].item()))
            cols.append(evecs[:, j] / t.sqrt(mass))
        return ks, cols, mass_ref

    def _pack_modes(self, k, phi_cols, mass, freq):
        """Assemble a :class:`ModeSet` from already self-consistent wavenumbers and eigenvector columns."""
        t = self._torch()
        del mass
        vecs = t.stack(phi_cols, dim=1) if phi_cols else t.zeros((self.n_z - 1, 0), dtype=t.float64)
        return self._finalize(k, vecs, freq)

    def _gamma2(self, omega, k):
        """Halfspace vertical decay rate ``gamma2 = sqrt(k^2 - (omega/c_bottom)^2)`` (0 if not trapped)."""
        g2sq = k**2 - (omega / self.c_bottom) ** 2
        return math.sqrt(g2sq) if g2sq > 0.0 else 0.0

    def _trapped_wavenumbers(self, evals, k_water, k_cut):
        t = self._torch()
        mask = (evals > k_cut**2) & (evals < k_water**2)
        return t.sqrt(evals[mask])

    def _assemble_modes(self, evals, evecs, mass, freq, k_water, k_cut):
        """Select the trapped eigenpairs (fixed-coefficient bottoms) and finalize them into a mode set."""
        t = self._torch()
        k2 = evals
        mask = (k2 > k_cut**2) & (k2 < k_water**2)
        idx = t.nonzero(mask, as_tuple=False).reshape(-1)
        # sort by descending k (mode 1 = highest wavenumber / lowest phase speed)
        order = idx[t.argsort(k2[idx], descending=True)]
        k = t.sqrt(k2[order])
        # undo the mass-lumping similarity transform: phi = M^{-1/2} y
        vecs = evecs[:, order] / t.sqrt(mass)[:, None]  # (n, n_mode)
        return self._finalize(k, vecs, freq)

    def _finalize(self, k, vecs, freq):
        """Rebuild full-depth mode shapes (dropped Dirichlet nodes zeroed), normalize to unit ``int phi^2 dz``,
        and fix the sign so the first interior node is positive."""
        t = self._torch()
        n_mode = int(k.numel())
        phi = t.zeros((self.n_z, n_mode), dtype=t.float64)
        if n_mode:
            if self.bottom == "pressure_release":
                phi[1 : self.n_z - 1, :] = vecs
            else:
                phi[1 : self.n_z, :] = vecs
            norm = t.sqrt(t.sum(phi * phi, dim=0) * self.dz)
            norm = t.where(norm > 0, norm, t.ones_like(norm))
            phi = phi / norm
            sign = t.sign(phi[1, :])
            sign = t.where(sign == 0, t.ones_like(sign), sign)
            phi = phi * sign
        return ModeSet(
            freq=float(freq),
            k=k.detach().cpu().numpy(),
            z=self.depths().detach().cpu().numpy(),
            phi=phi.detach().cpu().numpy(),
        )

    # ---- differentiable field synthesis -------------------------------------------------------------
    def field(self, modes: ModeSet, z_s: float, r, z=None):
        """Synthesize the complex modal-sum pressure ``p(r, z)`` from a solved :class:`ModeSet`.

        ``p(r, z) = ( i e^{-i pi/4} / (rho(z_s) sqrt(8 pi r)) ) sum_m phi_m(z_s) phi_m(z) e^{i k_m r} / sqrt(k_m)``,
        the far-field Hankel form of the mode expansion. ``z_s`` is the source depth (m), ``r`` a range or
        array of ranges (m), and ``z`` the receiver depths (defaults to the full depth axis). Returns a complex
        tensor of shape ``(n_r, n_z)`` (or ``(n_z,)`` for a scalar range). Differentiable in the modal data.
        """
        t = self._torch()
        k = t.as_tensor(modes.k, dtype=t.float64)
        z_axis = self.depths() if z is None else t.as_tensor(z, dtype=t.float64)
        phi_z = self._interp_modes(modes, z_axis)  # (n_z, n_mode)
        phi_s = self._interp_modes(modes, t.as_tensor([float(z_s)], dtype=t.float64))[0]  # (n_mode,)

        r_t = t.as_tensor(np.atleast_1d(np.asarray(r, dtype=float)), dtype=t.float64)  # (n_r,)
        prefac = (1j * math.e ** (-1j * math.pi / 4.0)) / (self.rho * t.sqrt(t.as_tensor(8.0 * math.pi)))
        # sum_m phi_s_m phi_z_m e^{i k_m r} / sqrt(k_m)
        phase = t.exp(1j * r_t[:, None] * k[None, :])  # (n_r, n_mode)
        amp = (phi_s / t.sqrt(k)).to(t.complex128)  # (n_mode,)
        modal = phase * amp[None, :]  # (n_r, n_mode)
        p = modal @ phi_z.to(t.complex128).T  # (n_r, n_z)
        p = prefac * p / t.sqrt(r_t.to(t.complex128))[:, None]
        return p[0] if np.ndim(r) == 0 else p

    def transmission_loss(self, modes: ModeSet, z_s: float, r, z=None, *, p_ref: float = 1.0):
        """Transmission loss ``TL = -20 log10(|p| / p_ref)`` (dB) of the modal-sum field."""
        t = self._torch()
        p = self.field(modes, z_s, r, z)
        return -20.0 * t.log10(t.abs(p) / p_ref + 1e-30)

    def _interp_modes(self, modes: ModeSet, z_query):
        """Linear interpolation of the mode shapes onto ``z_query`` (the modes live on the depth axis)."""
        t = self._torch()
        z0 = t.as_tensor(modes.z, dtype=t.float64)
        phi = t.as_tensor(modes.phi, dtype=t.float64)  # (n_z, n_mode)
        zq = t.clamp(z_query, float(z0[0]), float(z0[-1]))
        pos = (zq - z0[0]) / self.dz
        lo = t.clamp(pos.floor().long(), 0, self.n_z - 2)
        frac = (pos - lo.to(t.float64))[:, None]
        return phi[lo] * (1.0 - frac) + phi[lo + 1] * frac
