"""Range-marched split-step Fourier parabolic-equation propagator (Tappert standard PE).

The parabolic equation is the one-way, small-angle reduction of the Helmholtz equation that turns a
boundary-value problem in the range-depth plane into an *initial-value* march in range. Removing the rapid
carrier ``exp(i k0 r)`` from the acoustic/EM field, ``p = psi(z, r) exp(i k0 r)``, the standard PE is

    d psi / dr = i k0 ( sqrt(1 + X) - 1 ) psi,   X = (1/k0^2) d^2/dz^2 + (n(z, r)^2 - 1),

with ``n`` the (range-dependent) refractive index. The split-step Fourier scheme (Tappert 1977) advances
one range step ``dr`` by a Strang splitting of the two parts of ``X``: a half environmental phase in
depth, a full diffraction step applied exactly in vertical-wavenumber space, then a second half phase,

    psi <- exp( i k0 (n - 1) dr / 2 ) psi                                   (env half-step, z-space)
    psi <- IFFT[ exp( i dr ( sqrt(k0^2 - kz^2) - k0 ) ) FFT[ psi ] ]        (diffraction, kz-space)
    psi <- exp( i k0 (n - 1) dr / 2 ) psi                                   (env half-step, z-space)

Each factor is a pure phase, so the free march is unitary and conserves the vertical energy integral of
``|psi|^2``. The same operator serves underwater acoustics (index ``n = c0/c`` from a sound-speed profile)
and radar tropospheric propagation (``n`` from the modified refractivity ``M`` via ``n = 1 + M*1e-6``); a
trapping ``M`` profile ducts the field along the surface far past the geometric horizon.

The whole march is written on the ``ops`` namespace (``ops.fft``/``ops.ifft``/``ops.fftfreq`` and
elementwise phases), so it is differentiable in the index field ``n(z, r)`` and drops into
``inverse.Differential`` as a forward model. A pressure-release ocean surface is imposed by the image
method (odd symmetry about ``z = 0`` on a mirrored grid, forcing ``psi = 0`` at the surface); an absorbing
layer at the far boundary (a smooth amplitude taper, like the sponge/PML edge of ``wave_pml``) keeps the
finite grid from wrapping.

Reference: Tappert, "The parabolic approximation method", in *Wave Propagation and Underwater Acoustics*,
Lecture Notes in Physics 70 (1977); Jensen, Kuperman, Porter & Schmidt, *Computational Ocean Acoustics*
(the Lloyd-mirror PE benchmark); Levy, *Parabolic Equation Methods for Electromagnetic Wave Propagation*
(radar ducting).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from mixle_pde.ops import make_ops

__all__ = ["ParabolicEquation2D", "modified_refractivity_index", "lloyd_mirror_pressure"]


def modified_refractivity_index(M):
    """Refractive index ``n = 1 + M * 1e-6`` from the radar modified refractivity ``M``.

    ``M`` (M-units) folds the Earth-curvature correction into the refractivity so that a flat-earth PE
    reproduces spherical-earth propagation; a layer where ``dM/dz < 0`` traps energy (a surface duct).
    Elementwise and differentiable (plain arithmetic), so ``M(z, r)`` can be a latent driver.
    """
    return 1.0 + M * 1.0e-6


def lloyd_mirror_pressure(k, z_s, z, r):
    """Analytic Lloyd-mirror far-field pressure magnitude ``|p| = (2/r) |sin(k z_s z / r)|``.

    A point source at depth ``z_s`` under a pressure-release surface plus its negative image at ``-z_s``
    give ``|p|^2 ~ (4/r^2) sin^2(k z_s z / r)`` in the far field. Used as the published benchmark the PE
    march is checked against. ``k`` wavenumber, ``z`` receiver depth(s), ``r`` range.
    """
    import numpy as np

    return (2.0 / r) * np.abs(np.sin(k * z_s * z / r))


class ParabolicEquation2D:
    """A range-marched split-step Fourier standard-PE one-way propagator (differentiable in ``n``).

    The vertical grid has ``nz`` samples spaced ``dz`` covering physical depths ``0 .. (nz-1)*dz``. With
    ``surface="pressure_release"`` the field is odd-extended about ``z = 0`` onto a mirrored grid of size
    ``2*nz`` so that the diffraction FFT sees a wavenumber comb consistent with ``psi = 0`` at the surface
    (the image method); with ``surface="free"`` no image is imposed (radar / free upper boundary). An
    absorbing amplitude taper of ``absorb`` samples at the deep boundary (and, for the mirror grid, its
    reflection) prevents wrap-around.

    Parameters
    ----------
    nz : int
        Number of physical depth samples.
    dz : float
        Depth spacing (m).
    dr : float
        Range step (m).
    freq : float, optional
        Frequency (Hz); with ``c0`` sets ``k0 = 2 pi freq / c0``. Give this or ``k0``.
    k0 : float, optional
        Reference wavenumber (rad/m). Overrides ``freq``.
    c0 : float
        Reference sound speed / phase speed (m/s), the index reference ``n = c0/c``.
    surface : {"pressure_release", "free"}
        Upper boundary condition (image method vs none).
    absorb : int
        Width of the deep absorbing taper in samples.
    absorb_strength : float
        Peak attenuation of the absorbing taper per range step (0..1; larger absorbs faster).
    """

    def __init__(
        self,
        nz: int,
        *,
        dz: float,
        dr: float,
        freq: float | None = None,
        k0: float | None = None,
        c0: float = 1500.0,
        surface: str = "pressure_release",
        absorb: int = 32,
        absorb_strength: float = 2.0,
        ops: Any | None = None,
    ):
        if k0 is None:
            if freq is None:
                raise ValueError("give either k0 (rad/m) or freq (Hz).")
            k0 = 2.0 * math.pi * float(freq) / float(c0)
        if surface not in ("pressure_release", "free"):
            raise ValueError("surface must be 'pressure_release' or 'free'.")
        self.nz = int(nz)
        self.dz = float(dz)
        self.dr = float(dr)
        self.k0 = float(k0)
        self.c0 = float(c0)
        self.freq = freq
        self.surface = surface
        self.absorb = int(absorb)
        self.absorb_strength = float(absorb_strength)
        self.ops = ops if ops is not None else make_ops()
        self.mirror = surface == "pressure_release"
        # working grid size: doubled (odd extension) for the pressure-release image method
        self.ngrid = 2 * self.nz if self.mirror else self.nz
        t = self._torch()
        # vertical wavenumbers of the working grid (angular; ordered like fft: DC, +, -)
        kz = self.ops.fftfreq(self.ngrid, spacing=self.dz)
        # diffraction propagator over one full range step: exp(i dr (sqrt(k0^2 - kz^2) - k0)).
        # evanescent (kz > k0) modes get an imaginary sqrt -> real decay, the correct radiation condition.
        arg = t.as_tensor(self.k0**2, dtype=t.float64) - kz * kz
        root = t.sqrt(t.as_tensor(arg, dtype=t.complex128))
        self._diffract = t.exp(1j * self.dr * (root - self.k0))
        self._absorb_profile = self._build_absorb(t)

    def _torch(self):
        import torch

        return torch

    def depths(self):
        """The physical depth axis ``[0, dz, 2 dz, ..., (nz-1) dz]`` as a float64 tensor."""
        t = self._torch()
        return t.arange(self.nz, dtype=t.float64) * self.dz

    # --- grid helpers -------------------------------------------------------------------------------------
    def _build_absorb(self, t):
        """A smooth amplitude taper: 1 in the interior, cosine-tapering to <1 in the ``absorb`` deep samples
        (and its mirror reflection on the doubled grid), applied once per range step."""
        prof = t.ones(self.ngrid, dtype=t.float64)
        w = self.absorb
        if w <= 0:
            return prof
        idx = t.arange(w, dtype=t.float64)
        # attenuation factor per step in [taper_min, 1], quadratic ramp toward the boundary
        ramp = (idx / w) ** 2  # 0 at inner edge -> 1 at the boundary
        atten = t.exp(-self.absorb_strength * ramp * self.dr / self.dz)
        if self.mirror:
            # doubled grid layout: [physical 0..nz-1][mirror -nz..-1]; deep boundary is the join at index nz
            taper = t.flip(atten, dims=[0])  # strongest at index nz-1
            prof[self.nz - w : self.nz] = taper
            prof[self.nz : self.nz + w] = atten  # symmetric on the mirror side
        else:
            prof[self.nz - w : self.nz] = t.flip(atten, dims=[0])
        return prof

    def _embed_index(self, n_col, t):
        """Embed a physical index column ``n(z)`` (length nz) onto the working grid. For the mirror grid the
        index is even-extended (the medium is the same in the image), while the field is odd-extended."""
        n_col = t.as_tensor(n_col, dtype=t.float64)
        if not self.mirror:
            return n_col
        return t.cat([n_col, t.flip(n_col, dims=[0])])

    def _embed_field(self, psi_col, t):
        """Odd-extend a physical field column onto the mirror grid (image method), or pass through."""
        if not self.mirror:
            return psi_col
        return t.cat([psi_col, -t.flip(psi_col, dims=[0])])

    def _extract_field(self, psi_grid):
        """Physical (upper) half of a working-grid field column."""
        return psi_grid[: self.nz] if self.mirror else psi_grid

    # --- starter ------------------------------------------------------------------------------------------
    def starter(self, z_s: float, *, width: float | None = None):
        """A Gaussian (Thomson-Chapman) self-starter centred at source depth ``z_s``.

        ``psi0(z) = sqrt(k0) exp(-(k0^2/2)(z - z_s)^2)`` approximates a point source with a narrow angular
        spectrum matched to the PE aperture; ``width`` overrides the default depth width ``1/k0``. Returns a
        physical-length (``nz``) complex column.
        """
        t = self._torch()
        z = self.depths()
        sig = float(width) if width is not None else 1.0 / self.k0
        amp = math.sqrt(self.k0)
        g = amp * t.exp(-0.5 * ((z - float(z_s)) / sig) ** 2)
        return g.to(t.complex128)

    # --- one range step -----------------------------------------------------------------------------------
    def step(self, psi_grid, n_grid):
        """Advance one working-grid field column ``psi_grid`` by one range step ``dr`` (Strang split-step).

        ``n_grid`` is the working-grid index column at this range. Returns the advanced column. Operates on
        the doubled grid directly (both are working-grid length); use :meth:`march` for the physical API.
        """
        t = self._torch()
        env_half = t.exp(0.5j * self.k0 * (n_grid.to(t.complex128) - 1.0) * self.dr)
        psi = env_half * psi_grid
        psi = self.ops.ifft(self._diffract * self.ops.fft(psi))
        psi = env_half * psi
        psi = psi * self._absorb_profile.to(t.complex128)
        return psi

    def march(self, psi0, n, n_range: int | None = None):
        """March a physical starter ``psi0`` (length nz) over range, returning ``psi`` at every range step.

        ``n`` is the refractive-index field: either a fixed physical column (length nz, range-independent)
        or a callable ``n(step) -> column`` of length nz for a range-dependent medium, or a 2-D array/tensor
        of shape ``(n_range, nz)``. Returns a complex tensor of shape ``(n_range, nz)`` with the physical
        field at ranges ``dr, 2 dr, ..., n_range dr`` (the starter itself is not included).
        """
        t = self._torch()
        psi = self._embed_field(t.as_tensor(psi0, dtype=t.complex128), t)
        n_fn = self._as_index_fn(n, t)
        if n_range is None:
            n_range = self._infer_n_range(n)
        cols = []
        for i in range(int(n_range)):
            n_grid = self._embed_index(n_fn(i), t)
            psi = self.step(psi, n_grid)
            cols.append(self._extract_field(psi))
        return t.stack(cols, dim=0)

    # --- pressure / transmission loss ---------------------------------------------------------------------
    def pressure(self, psi_field, ranges=None):
        """Complex acoustic pressure ``p = psi / sqrt(r)`` from the marched envelope field.

        The PE envelope drops the cylindrical-spreading and carrier factors; the physical pressure is
        ``p(z, r) = psi(z, r) exp(i k0 r) / sqrt(r)``. Only the magnitude structure matters for the Lloyd
        benchmark and transmission loss, so the returned complex field carries the ``1/sqrt(r)`` spreading
        and the carrier phase. ``psi_field`` is ``(n_range, nz)``; ``ranges`` defaults to ``dr..n_range dr``.
        """
        t = self._torch()
        nr = psi_field.shape[0]
        if ranges is None:
            ranges = (t.arange(nr, dtype=t.float64) + 1.0) * self.dr
        else:
            ranges = t.as_tensor(ranges, dtype=t.float64)
        r = ranges.reshape(-1, 1).to(t.complex128)
        carrier = t.exp(1j * self.k0 * ranges).reshape(-1, 1)
        return psi_field * carrier / t.sqrt(r)

    def transmission_loss(self, psi_field, ranges=None, *, p_ref: float = 1.0):
        """Transmission loss ``TL = -20 log10(|p| / p_ref)`` (dB) from the marched field."""
        t = self._torch()
        p = self.pressure(psi_field, ranges)
        return -20.0 * t.log10(t.abs(p) / p_ref + 1e-30)

    def energy(self, psi_field):
        """Vertical energy integral ``sum |psi|^2 dz`` per range (conserved in a lossless section)."""
        t = self._torch()
        return t.sum(t.abs(psi_field) ** 2, dim=1) * self.dz

    # --- index-field plumbing -----------------------------------------------------------------------------
    def _as_index_fn(self, n, t) -> Callable[[int], Any]:
        if callable(n):
            return lambda i: t.as_tensor(n(i), dtype=t.float64)
        arr = t.as_tensor(n, dtype=t.float64)
        if arr.dim() == 1:
            return lambda i: arr
        return lambda i: arr[i]

    @staticmethod
    def _infer_n_range(n) -> int:
        try:
            import torch

            if torch.is_tensor(n) and n.dim() == 2:
                return int(n.shape[0])
        except ImportError:
            pass
        if getattr(n, "ndim", None) == 2:
            return int(n.shape[0])
        raise ValueError("give n_range explicitly for a range-independent or callable index field.")
