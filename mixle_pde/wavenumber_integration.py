"""Full-wave field in a horizontally-stratified medium by wavenumber integration (the Fast-Field method).

The OASES / SCOOTER reference forward for a range-independent (horizontally-stratified) ocean: a point source
in a stack of homogeneous fluid layers over a fluid halfspace. Because the medium depends only on depth, a
Hankel transform in range separates the 3-D Helmholtz equation into an independent 1-D depth problem at every
horizontal wavenumber ``kr``. In each layer the depth Green's function is a pair of analytic plane waves
``exp(+/- i kz z)`` with ``kz^2 = k(z)^2 - kr^2``; matching pressure and normal particle velocity ``(1/rho) dp/dz``
across the interfaces (and the pressure-release surface + radiation into the bottom halfspace) fixes them. The
field is then resynthesized by the inverse Hankel transform

    p(r, z) = integral_0^inf g(kr, z; zs) J0(kr r) kr dkr,

evaluated here as a direct Bessel quadrature over a real ``kr`` grid. This is the full-wave answer the faster
parabolic-equation and normal-mode forwards are validated against: it carries the discrete trapped modes AND the
continuous spectrum (the branch-line / lateral-wave field) that a mode sum drops.

The depth solve is the Wronskian construction of the 1-D Green's function: a homogeneous solution ``u_lo`` that
satisfies the surface condition and one ``u_hi`` that satisfies the bottom radiation condition, combined as
``g = u_lo(z_<) u_hi(z_>) / W`` with the (rho-weighted) Wronskian ``W``. Attenuation enters through a complex
sound speed ``c(1 + i beta)`` (equivalently a complex wavenumber), which also moves the trapped-mode poles off
the real ``kr`` axis so the quadrature is well behaved. A layered elastic seabed is out of scope here (see
:mod:`mixle_pde.elastic` / :mod:`mixle_pde.poroelastic` for the layered moduli); the bottom is fluid.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j0

__all__ = ["WavenumberIntegration1D"]


def _torch():
    import torch

    return torch


class WavenumberIntegration1D:
    """Wavenumber-integration (Fast-Field) full-wave field for a stratified fluid ocean.

    ``WavenumberIntegration1D(freq, depths, speeds, densities, ...)`` builds the forward for a stack of
    homogeneous fluid layers over a fluid halfspace, driven by a point source at depth ``zs``. ``depths`` are
    the interface depths ``z_1 < z_2 < ... < z_{L}`` (metres below the pressure-release surface at ``z = 0``);
    ``speeds`` and ``densities`` list the ``L`` layer values plus one halfspace value (length ``L + 1``). The
    top of the water column is a pressure-release surface (``p = 0`` at ``z = 0``) and the bottom halfspace
    radiates downward.

    A single homogeneous layer over the same-medium halfspace reduces to free space, and the wavenumber
    integral then reproduces the Sommerfeld point-source Green's function ``exp(i k R) / (4 pi R)``
    (:meth:`green` / :meth:`field`). A slow-over-fast pair is a Pekeris waveguide whose transmission loss shows
    modal interference on cylindrical spreading.

    Attenuation is set per layer by ``beta`` (a complex-speed fraction ``c -> c (1 + i beta)``); a small nonzero
    ``beta`` also regularizes the trapped-mode poles for the real-axis quadrature. The horizontal-wavenumber
    grid is ``kr in (0, kr_max]`` with ``kr_max = kr_max_fac * omega / min(Re c)`` and ``n_kr`` points.
    """

    def __init__(
        self,
        freq: float,
        depths,
        speeds,
        densities,
        *,
        zs: float,
        beta: float | np.ndarray = 0.0,
        n_kr: int = 200_000,
        kr_max_fac: float = 4.0,
        surface: bool = True,
    ):
        self.freq = float(freq)
        self.omega = 2.0 * np.pi * self.freq
        self.zs = float(zs)
        self.surface = bool(surface)

        self.depths = np.asarray(depths, dtype=float).reshape(-1)  # interface depths, length L
        self.speeds = np.asarray(speeds, dtype=float).reshape(-1)  # length L + 1 (layers + halfspace)
        self.densities = np.asarray(densities, dtype=float).reshape(-1)
        n_media = self.speeds.size
        if self.depths.size != n_media - 1:
            raise ValueError(f"expected len(depths) == len(speeds) - 1; got {self.depths.size} and {self.speeds.size}.")
        if self.densities.size != n_media:
            raise ValueError(f"expected len(densities) == len(speeds) == {n_media}; got {self.densities.size}.")

        beta = np.broadcast_to(np.asarray(beta, dtype=float), (n_media,)).astype(float)
        # complex wavenumbers per medium: k = (omega / c) (1 + i beta). With the exp(-i omega t) / exp(+i k R)
        # outgoing convention this Im(k) > 0 gives amplitude decay exp(-omega beta R / c).
        self.k = (self.omega / self.speeds) * (1.0 + 1j * beta)  # complex, length n_media

        cmin = float(np.min(self.speeds))
        kr_max = kr_max_fac * self.omega / cmin
        # skip kr = 0 (J0 fine there, but kz singular for the marginal ray); start just above 0
        self.kr = np.linspace(kr_max / n_kr, kr_max, int(n_kr))

    # ---- per-kr depth Green's function --------------------------------------------------------------
    def _vertical_wavenumbers(self, torch, kr):
        """Vertical wavenumbers ``kz = sqrt(k^2 - kr^2)`` per medium, branch chosen for decay (Im kz >= 0)."""
        k = torch.as_tensor(self.k, dtype=torch.complex128)
        kr2 = torch.as_tensor(kr**2, dtype=torch.complex128)
        # kz[m, j]: medium m, wavenumber-grid point j
        kz = torch.sqrt(k[:, None] ** 2 - kr2[None, :])
        kz = torch.where(kz.imag < 0, -kz, kz)  # decaying / outgoing branch
        return kz

    def green(self, z_rcv, *, torch=None):
        """Depth Green's function ``g(kr, z_rcv; zs)`` at every ``kr`` on the grid (complex, length ``n_kr``).

        Solves the 1-D depth problem ``(d^2/dz^2 + kz^2) g = -delta(z - zs)`` through the layered stack by the
        Wronskian construction: ``u_lo`` carries the pressure-release surface condition down from ``z = 0``,
        ``u_hi`` carries the bottom radiation condition up from the halfspace, both propagated across interfaces
        with continuity of pressure and of ``(1/rho) dp/dz``. Returns ``g = u_lo(z_<) u_hi(z_>) / W`` with the
        rho-weighted Wronskian ``W``. Differentiable in the medium wavenumbers through ``torch``.

        Args:
            z_rcv: receiver depth (m below the surface).
            torch: backend module (defaults to the package torch import).

        Returns:
            complex torch tensor ``g`` of length ``n_kr``.
        """
        torch = torch or _torch()
        kz = self._vertical_wavenumbers(torch, self.kr)  # (n_media, n_kr)
        rho = torch.as_tensor(self.densities, dtype=torch.complex128)
        depths = self.depths
        n_media = self.speeds.size
        z_rcv = float(z_rcv)
        zs = self.zs

        # layer index containing a depth z (0..n_media-1; last is the halfspace)
        def _layer_of(z):
            return int(np.searchsorted(depths, z, side="right"))

        # --- u_lo: pressure-release surface p(0) = 0, propagated down through the layers -----------------
        # In layer m, u_lo = A_m e^{i kz_m (z - z_top_m)} + B_m e^{-i kz_m (z - z_top_m)} referenced to the
        # layer top. With a pressure-release surface, u_lo = sin(kz_0 z) => A_0 = 1/(2i), B_0 = -1/(2i). With
        # no surface (a truly infinite upper halfspace) u_lo is the upgoing radiation e^{-i kz_0 z}, which
        # decays as z -> -inf; the field then reduces to free space.
        if self.surface:
            A = torch.full_like(kz[0], 0.5 / 1j)
            B = -A
        else:
            A = torch.zeros_like(kz[0])
            B = torch.ones_like(kz[0])

        def _uval(A, B, kzm, dz):
            e = torch.exp(1j * kzm * dz)
            return A * e + B / e

        def _udrv(A, B, kzm, dz):
            e = torch.exp(1j * kzm * dz)
            return 1j * kzm * (A * e - B / e)

        # walk down to the layer holding zs, storing the layer-top coefficients we pass through
        lo_layers = [(0.0, A, B)]  # (z_top, A, B) per layer traversed
        for m in range(n_media - 1):
            zt = 0.0 if m == 0 else float(depths[m - 1])
            zb = float(depths[m])
            dz = zb - zt
            # value + derivative at the bottom interface of layer m
            val = _uval(A, B, kz[m], dz)
            drv = _udrv(A, B, kz[m], dz)
            # continuity of pressure and (1/rho) dp/dz -> coefficients of layer m+1 at its top (zb)
            kzn = kz[m + 1]
            # val = A' + B' ; (1/rho_{m+1}) drv_next = (1/rho_m) drv ; drv_next = i kzn (A' - B')
            rhs_drv = (rho[m + 1] / rho[m]) * drv
            Ap = 0.5 * (val + rhs_drv / (1j * kzn))
            Bp = 0.5 * (val - rhs_drv / (1j * kzn))
            A, B = Ap, Bp
            lo_layers.append((zb, A, B))

        def u_lo(z):
            m = _layer_of(z)
            zt, Am, Bm = lo_layers[m]
            return _uval(Am, Bm, kz[m], z - zt)

        def u_lo_p(z):
            m = _layer_of(z)
            zt, Am, Bm = lo_layers[m]
            return _udrv(Am, Bm, kz[m], z - zt)

        # --- u_hi: bottom radiation condition, propagated up through the layers --------------------------
        # In the halfspace (medium n_media-1) u_hi = e^{i kz_h (z - z_H)} (downgoing, decaying). Reference it
        # to the last interface z_H = depths[-1]; at z_H value 1, derivative i kz_h.
        n_last = n_media - 1
        if depths.size == 0:
            # single homogeneous medium: free space, upper solution is e^{i kz z}
            z_H = 0.0
            Ah = torch.ones_like(kz[0])
            Bh = torch.zeros_like(kz[0])
            hi_layers = {0: (0.0, Ah, Bh)}
        else:
            z_H = float(depths[-1])
            # halfspace coefficients referenced to z_H
            Ah = torch.ones_like(kz[n_last])
            Bh = torch.zeros_like(kz[n_last])
            hi_layers = {n_last: (z_H, Ah, Bh)}
            # walk up: at each interface match into the layer above (reference each layer to ITS top)
            A_i, B_i = Ah, Bh  # coeffs of current (lower) medium referenced to interface z_i
            for m in range(n_last, 0, -1):
                z_i = float(depths[m - 1])  # top interface of medium m
                # value + derivative of the lower medium's solution at z_i (its own reference is z_i for m<n_last,
                # but halfspace/medium referenced to z_H): recompute referenced to z_i
                zt_m = hi_layers[m][0]
                val = _uval(A_i, B_i, kz[m], z_i - zt_m)
                drv = _udrv(A_i, B_i, kz[m], z_i - zt_m)
                kzu = kz[m - 1]
                rhs_drv = (rho[m - 1] / rho[m]) * drv
                Au = 0.5 * (val + rhs_drv / (1j * kzu))
                Bu = 0.5 * (val - rhs_drv / (1j * kzu))
                hi_layers[m - 1] = (z_i, Au, Bu)
                A_i, B_i = Au, Bu

        def u_hi(z):
            m = _layer_of(z)
            zt, Am, Bm = hi_layers[m]
            return _uval(Am, Bm, kz[m], z - zt)

        def u_hi_p(z):
            m = _layer_of(z)
            zt, Am, Bm = hi_layers[m]
            return _udrv(Am, Bm, kz[m], z - zt)

        # --- Wronskian at the source and the two-sided Green's function ---------------------------------
        # W = u_lo u_hi' - u_lo' u_hi (rho-weighting cancels since both derivatives are already p/rho matched
        # in the same source layer). Evaluate at zs.
        rho_s = rho[_layer_of(zs)]
        W = (u_lo(zs) * u_hi_p(zs) - u_lo_p(zs) * u_hi(zs)) / rho_s
        z_lt = min(z_rcv, zs)
        z_gt = max(z_rcv, zs)
        # -1/(rho_s W): the delta-source has strength 1/rho in the (1/rho) p' formulation. The 1/(2 pi) is the
        # inverse-Hankel normalization, so integral g J0 kr dkr = exp(i k R)/(4 pi R) in free space.
        g = -u_lo(z_lt) * u_hi(z_gt) / (rho_s * W) / (2.0 * np.pi)
        return g

    # ---- range synthesis (inverse Hankel transform) -------------------------------------------------
    def field(self, r, z_rcv, *, torch=None):
        """Synthesize the complex pressure ``p(r, z)`` by the inverse Hankel transform (Bessel quadrature).

        Evaluates ``p(r, z) = integral_0^inf g(kr, z; zs) J0(kr r) kr dkr`` on the ``kr`` grid with the
        trapezoid rule. ``r`` may be a scalar or a 1-D array of ranges (all synthesized from one depth solve).
        The Bessel weights are real data; the medium dependence rides on ``g``, so the map stays differentiable
        in the medium wavenumbers through ``torch``.

        Args:
            r: source-receiver range(s) in metres (scalar or 1-D array).
            z_rcv: receiver depth (m).
            torch: backend module (defaults to the package torch import).

        Returns:
            complex torch tensor ``p`` (scalar shape if ``r`` scalar, else shape ``(len(r),)``).
        """
        torch = torch or _torch()
        g = self.green(z_rcv, torch=torch)  # (n_kr,)
        kr = self.kr
        r_arr = np.atleast_1d(np.asarray(r, dtype=float))
        # integrand weight kr * dkr per node (trapezoid); grid is uniform
        dkr = float(kr[1] - kr[0])
        weight = torch.as_tensor(kr * dkr, dtype=torch.complex128)
        w_g = g * weight  # (n_kr,)
        out = []
        for rv in r_arr:
            bessel = torch.as_tensor(j0(kr * float(rv)), dtype=torch.complex128)
            out.append(torch.sum(w_g * bessel))
        p = torch.stack(out)
        return p[0] if np.isscalar(r) or np.asarray(r).ndim == 0 else p

    def transmission_loss(self, r, z_rcv, *, torch=None):
        """Transmission loss ``TL = -20 log10 |p(r, z)|`` (dB re 1 m) at the given range(s) and depth."""
        torch = torch or _torch()
        p = self.field(r, z_rcv, torch=torch)
        return -20.0 * torch.log10(torch.abs(p) + 1e-30)

    # ---- reference solutions ------------------------------------------------------------------------
    def sommerfeld(self, r, z_rcv):
        """Free-space point-source Green's function ``exp(i k R) / (4 pi R)``, ``R = sqrt(r^2 + (z - zs)^2)``.

        The analytic target for the single-halfspace recovery check (uses the top-medium wavenumber, real part
        for the phase; a lossless medium gives a pure outgoing spherical wave).
        """
        r_arr = np.atleast_1d(np.asarray(r, dtype=float))
        k = complex(self.k[0])
        R = np.sqrt(r_arr**2 + (z_rcv - self.zs) ** 2)
        p = np.exp(1j * k * R) / (4.0 * np.pi * R)
        return p[0] if np.isscalar(r) or np.asarray(r).ndim == 0 else p

    def mode_wavenumber_window(self):
        """The trapped-mode horizontal-wavenumber band ``[omega / c_bottom, omega / c_water_min]``.

        Real horizontal wavenumbers of the discrete waveguide modes lie in this interval: below
        ``omega / c_bottom`` the field radiates into the halfspace (continuous spectrum), above
        ``omega / c_water_min`` it is evanescent in the water.
        """
        c_water = float(np.min(self.speeds[:-1])) if self.speeds.size > 1 else float(self.speeds[0])
        c_bottom = float(self.speeds[-1])
        return self.omega / c_bottom, self.omega / c_water
