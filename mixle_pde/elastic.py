"""3D isotropic elastodynamic wave equation on a staggered grid, explicit velocity-stress leapfrog.

The physics behind seismic ground motion and full-waveform inversion: the elastic wave equation

    rho d^2 u/dt^2 = (lambda + mu) grad(div u) + mu laplacian(u) + f

rewritten as the first-order velocity-stress system (Virieux, 1986) and integrated by an explicit
leapfrog in which the particle velocity ``v = du/dt`` (a 3-vector) and the symmetric stress tensor
``sigma`` (six independent components) live a half time-step apart:

    dv/dt     = (1/rho) div(sigma) + source
    dsigma/dt = C : grad(v)

with the isotropic stiffness ``C``: ``dsigma_xx = (lambda+2mu) dvx/dx + lambda (dvy/dy + dvz/dz)`` and
cyclic permutations for the normal stresses, ``dsigma_xy = mu (dvx/dy + dvy/dx)`` and cyclic for the shear
stresses. The medium carries two wave speeds -- a compressional (P) wave at ``vp = sqrt((lambda+2mu)/rho)``
and a shear (S) wave at ``vs = sqrt(mu/rho)`` -- so the same solver produces both body waves and, with a
free surface, the surface waves that dominate a seismogram. The parameters may be scalars or full fields
(a heterogeneous velocity model), with ``lambda = rho (vp^2 - 2 vs^2)`` and ``mu = rho vs^2``.

The staggering is the elastic analog of the Yee cell in :class:`~mixle_pde.maxwell.Maxwell3D`: the three
velocity components and six stress components sit at half-integer offsets so every derivative in the update
is a centred difference at the point where the update lives. The same forward/backward one-sided differences
(``_dp``/``_dm``) that centre the EM curls centre the elastic gradient and divergence here. An absorbing
sponge near the sides damps the velocity so the finite box does not reflect outgoing waves (the sponge idea
of :class:`~mixle_pde.wave.WaveEquation2D` and :class:`~mixle_pde.wave3d.WaveEquation3D`, extended to a
vector velocity), and an optional traction-free top boundary (``sigma_zz = sigma_xz = sigma_yz = 0``) is the
free surface that launches Rayleigh surface waves.

The stepper matches the wave/maxwell API: ``ElasticWave3D(n, dt=..., spacing=..., vp=..., vs=..., rho=...)``
builds the forward on an ``n x n x n`` grid; ``pack``/``unpack`` move between the flat integrator state and
the nine field arrays; ``step(state, ops, source=...)`` advances one full leapfrog step. It is
differentiable through ``ops`` (slice arithmetic on autograd tensors), so a seismic observation is a
:class:`~mixle_pde.inverse.Differential` forward like the acoustic wave, and the gradient w.r.t. the
velocity model is the full-waveform-inversion sensitivity.
"""

from __future__ import annotations

import numpy as np

# component order in the packed state: three velocities then six stresses (symmetric tensor)
_VEL = ("vx", "vy", "vz")
_STRESS = ("sxx", "syy", "szz", "sxy", "sxz", "syz")
_COMPONENTS = _VEL + _STRESS


class ElasticWave3D:
    """A differentiable 3D isotropic elastic-wave stepper (velocity-stress staggered leapfrog).

    ``ElasticWave3D(n, dt=..., spacing=..., vp=..., vs=..., rho=...)`` builds the forward on an ``n x n x n``
    grid. Pass the wave speeds ``vp, vs`` and density ``rho`` (scalars or ``n^3`` fields) or, equivalently,
    the Lame parameters ``lam, mu`` and ``rho``; the two are related by ``lam = rho (vp^2 - 2 vs^2)`` and
    ``mu = rho vs^2``. The state packs the nine components in the order
    ``(vx, vy, vz, sxx, syy, szz, sxy, sxz, syz)`` (``pack``/``unpack``); ``step(state, ops, source=...)``
    advances one full leapfrog step (stress a half step, then velocity a half step).

    Staggering (component -> half-integer offset from the ``(i,j,k)`` node, in units of ``spacing``):
    the diagonal stresses ``sxx, syy, szz`` at the node ``(i,j,k)``; ``vx@(i+1/2,j,k)``, ``vy@(i,j+1/2,k)``,
    ``vz@(i,j,k+1/2)``; and the shear stresses on the cell edges, ``sxy@(i+1/2,j+1/2,k)``,
    ``sxz@(i+1/2,j,k+1/2)``, ``syz@(i,j+1/2,k+1/2)``. Each derivative in the update is then a one-sided
    difference of the dual field between the two staggered points straddling the update location, i.e. a
    centred difference there. The 3D Courant limit is ``dt <= spacing / (vp * sqrt(3))``.
    """

    def __init__(
        self,
        n: int,
        *,
        dt: float,
        spacing: float | None = None,
        vp: float | np.ndarray = 1.7,
        vs: float | np.ndarray = 1.0,
        rho: float | np.ndarray = 1.0,
        lam: float | np.ndarray | None = None,
        mu: float | np.ndarray | None = None,
        absorb_width: int = 0,
        absorb_strength: float = 2.0,
        free_surface: bool = False,
    ):
        self.n = int(n)
        self.dt = float(dt)
        self.h = float(spacing) if spacing is not None else 1.0 / (n - 1)
        self._nc = self.n**3  # per-component length
        self.free_surface = bool(free_surface)

        rho = self._as_field(rho)
        if lam is not None and mu is not None:
            lam = self._as_field(lam)
            mu = self._as_field(mu)
        else:
            vp = self._as_field(vp)
            vs = self._as_field(vs)
            lam = rho * (vp**2 - 2.0 * vs**2)
            mu = rho * vs**2
        self.rho = rho
        self.lam = lam
        self.mu = mu
        # peak P speed sets the Courant number reported to the user
        self.vp_max = float(np.sqrt((np.max(lam) + 2.0 * np.max(mu)) / np.min(rho)))
        self._inv_rho = 1.0 / rho
        self._gamma = self._build_sponge(absorb_width, absorb_strength)

    def _as_field(self, x):
        """Broadcast a scalar or array parameter to an ``(n, n, n)`` field."""
        a = np.asarray(x, dtype=float)
        if a.ndim == 0:
            return np.full((self.n, self.n, self.n), float(a))
        return a.reshape(self.n, self.n, self.n).astype(float)

    def _build_sponge(self, width, strength):
        """A velocity-damping ramp near the six faces (or the four sides + bottom under a free surface)."""
        n = self.n
        gamma = np.zeros((n, n, n))
        if width > 0:
            idx = np.arange(n)
            d = np.minimum(idx, n - 1 - idx)  # distance (in nodes) to the nearest edge
            ramp = np.where(d < width, (1.0 - d / width) ** 2, 0.0)
            gx = np.broadcast_to(ramp[:, None, None], (n, n, n))
            gy = np.broadcast_to(ramp[None, :, None], (n, n, n))
            gz = np.broadcast_to(ramp[None, None, :], (n, n, n))
            if self.free_surface:
                # do not damp toward the free top face (z = 0); surface waves must live there
                top = np.where(idx < width, (1.0 - idx / width) ** 2, 0.0)
                gz_top = np.broadcast_to(top[None, None, :], (n, n, n))
                gz = np.maximum(gz - gz_top, 0.0)
            gamma = float(strength) * np.maximum(np.maximum(gx, gy), gz)
        return gamma.ravel()

    # ---- state packing -------------------------------------------------------------------------------
    def pack(self, vx, vy, vz, sxx, syy, szz, sxy, sxz, syz):
        """Pack the nine field arrays (each ``n x n x n`` or flat) into the integrator state."""
        import torch

        parts = [torch.as_tensor(f).reshape(-1) for f in (vx, vy, vz, sxx, syy, szz, sxy, sxz, syz)]
        return torch.cat(parts)

    def unpack(self, state):
        """Split a packed state into the nine components as flat length-``n^3`` tensors, in order."""
        nc = self._nc
        return tuple(state[i * nc : (i + 1) * nc] for i in range(9))

    def fields(self, state, ops):
        """Return the nine components reshaped to ``(n, n, n)`` grids (a view for accessors / sampling)."""
        n = self.n
        return tuple(c.reshape(n, n, n) for c in self.unpack(state))

    def velocity(self, state, ops):
        """The three particle-velocity components ``(vx, vy, vz)`` as ``(n, n, n)`` grids."""
        n = self.n
        vx, vy, vz = (c.reshape(n, n, n) for c in self.unpack(state)[:3])
        return vx, vy, vz

    def zeros(self, ops):
        """A zero field state (all nine components) as a packed integrator state."""
        return ops.zeros(9 * self._nc)

    # ---- staggered-grid one-sided differences ---------------------------------------------------------
    # Forward difference d_a f: (f[i+1] - f[i]) / h centred a half cell forward of f along axis a.
    @staticmethod
    def _dp(f, axis, h):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        out = f * 0.0
        out[tuple(lo)] = (f[tuple(hi)] - f[tuple(lo)]) / h
        return out

    # Backward difference d_a f: (f[i] - f[i-1]) / h centred a half cell backward of f along axis a.
    @staticmethod
    def _dm(f, axis, h):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(1, None)
        hi[axis] = slice(0, -1)
        out = f * 0.0
        out[tuple(lo)] = (f[tuple(lo)] - f[tuple(hi)]) / h
        return out

    # ---- leapfrog step -------------------------------------------------------------------------------
    def step(self, state, ops, *, source=None):
        """Advance one full velocity-stress leapfrog step (update stress, then velocity).

        ``source`` is an optional dict of per-component additive rate injections for this step, keyed by
        component name (``"vx"``..``"szz"``); use :meth:`moment_tensor_source` or :meth:`point_force_source`
        to build one. The stress update sees the pre-update velocity; the velocity update sees the just-
        updated stress (the standard staggered leapfrog ordering). The velocity is damped by the sponge and,
        with ``free_surface=True``, the traction-free top boundary is enforced on the stresses each step.
        """
        n, h, dt = self.n, self.h, self.dt
        vx, vy, vz, sxx, syy, szz, sxy, sxz, syz = (c.reshape(n, n, n) for c in self.unpack(state))
        lam = ops.tensor(self.lam)
        mu = ops.tensor(self.mu)
        inv_rho = ops.tensor(self._inv_rho)
        src = source or {}

        def add(name, field):
            s = src.get(name)
            return field if s is None else field + dt * ops.tensor(np.asarray(s).reshape(n, n, n))

        # --- stress update: sigma^{q+1/2} = sigma^{q-1/2} + dt (C : grad v^q) --------------------------
        # velocity gradients centred at the stress locations
        dvx_dx = self._dm(vx, 0, h)  # vx@(i+1/2) back-diff -> node (i)
        dvy_dy = self._dm(vy, 1, h)
        dvz_dz = self._dm(vz, 2, h)
        div_v = dvx_dx + dvy_dy + dvz_dz
        lam2mu = lam + 2.0 * mu
        sxx = add("sxx", sxx + dt * (lam2mu * dvx_dx + lam * (dvy_dy + dvz_dz)))
        syy = add("syy", syy + dt * (lam2mu * dvy_dy + lam * (dvx_dx + dvz_dz)))
        szz = add("szz", szz + dt * (lam2mu * dvz_dz + lam * (dvx_dx + dvy_dy)))

        # shear-stress gradients centred at the edge locations (forward diff of the dual velocity)
        dvx_dy = self._dp(vx, 1, h)  # vx@(i+1/2) fwd-diff in y -> edge (i+1/2, j+1/2)
        dvy_dx = self._dp(vy, 0, h)
        dvx_dz = self._dp(vx, 2, h)
        dvz_dx = self._dp(vz, 0, h)
        dvy_dz = self._dp(vy, 2, h)
        dvz_dy = self._dp(vz, 1, h)
        # shear modulus averaged to the edge where the shear stress lives (harmonic-free simple mean)
        mu_xy = 0.5 * (mu + self._roll_avg(mu, (0, 1)))
        mu_xz = 0.5 * (mu + self._roll_avg(mu, (0, 2)))
        mu_yz = 0.5 * (mu + self._roll_avg(mu, (1, 2)))
        sxy = add("sxy", sxy + dt * mu_xy * (dvx_dy + dvy_dx))
        sxz = add("sxz", sxz + dt * mu_xz * (dvx_dz + dvz_dx))
        syz = add("syz", syz + dt * mu_yz * (dvy_dz + dvz_dy))

        if self.free_surface:
            sxx, syy, szz, sxy, sxz, syz = self._apply_free_surface(sxx, syy, szz, sxy, sxz, syz)

        # --- velocity update: v^{q+1} = v^q + dt (1/rho) div sigma^{q+1/2} + source ---------------------
        # stress divergence centred at each velocity location
        # (div sigma)_x at vx@(i+1/2): d sxx/dx (fwd), d sxy/dy (back), d sxz/dz (back)
        fx = self._dp(sxx, 0, h) + self._dm(sxy, 1, h) + self._dm(sxz, 2, h)
        fy = self._dm(sxy, 0, h) + self._dp(syy, 1, h) + self._dm(syz, 2, h)
        fz = self._dm(sxz, 0, h) + self._dm(syz, 1, h) + self._dp(szz, 2, h)
        vx = add("vx", vx + dt * inv_rho * fx)
        vy = add("vy", vy + dt * inv_rho * fy)
        vz = add("vz", vz + dt * inv_rho * fz)

        # absorbing sponge: damp the velocity toward zero near the sides
        gamma = ops.tensor(self._gamma).reshape(n, n, n)
        damp = 1.0 / (1.0 + dt * gamma)
        vx = vx * damp
        vy = vy * damp
        vz = vz * damp

        return ops.cat(
            [
                vx.reshape(-1),
                vy.reshape(-1),
                vz.reshape(-1),
                sxx.reshape(-1),
                syy.reshape(-1),
                szz.reshape(-1),
                sxy.reshape(-1),
                sxz.reshape(-1),
                syz.reshape(-1),
            ]
        )

    @staticmethod
    def _roll_avg(f, axes):
        """The value of ``f`` shifted forward by one node along each of ``axes`` (for edge averaging)."""
        out = f
        for a in axes:
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[a] = slice(0, -1)
            hi[a] = slice(1, None)
            shifted = out * 0.0
            shifted[tuple(lo)] = out[tuple(hi)]
            out = shifted
        return out

    # ---- free surface --------------------------------------------------------------------------------
    def _apply_free_surface(self, sxx, syy, szz, sxy, sxz, syz):
        """Traction-free top boundary (z = 0): the tractions on the top face vanish.

        The outward normal of the top face is along z, so the three tractions ``sigma_zz, sigma_xz,
        sigma_yz`` must be zero there. Held on the ``(n,n,n)`` component grids at ``k = 0``; the resulting
        impedance jump reflects body waves and traps the Rayleigh surface wave along the top.
        """
        szz[:, :, 0] = 0.0
        sxz[:, :, 0] = 0.0
        syz[:, :, 0] = 0.0
        return sxx, syy, szz, sxy, sxz, syz

    # ---- point sources -------------------------------------------------------------------------------
    def moment_tensor_source(self, position, moment, amplitude=1.0):
        """A moment-tensor point source at ``position`` -- a stress-rate injection that excites P and S.

        ``moment`` is the symmetric seismic moment tensor ``M`` (a 3x3 array or the six independent
        components ``[Mxx, Myy, Mzz, Mxy, Mxz, Myz]``). An explosion is ``M = I`` (pure P), a double couple
        (e.g. ``Mxz = Mzx``) radiates the four-lobed S pattern of an earthquake. Returns a ``source`` dict
        for :meth:`step` that adds ``amplitude * M`` to the stress rate at the source node for that step.
        """
        i, j, k = (int(round(p)) for p in position)
        m = np.asarray(moment, dtype=float)
        if m.shape == (3, 3):
            comps = {
                "sxx": m[0, 0],
                "syy": m[1, 1],
                "szz": m[2, 2],
                "sxy": 0.5 * (m[0, 1] + m[1, 0]),
                "sxz": 0.5 * (m[0, 2] + m[2, 0]),
                "syz": 0.5 * (m[1, 2] + m[2, 1]),
            }
        else:
            m = m.reshape(6)
            comps = dict(zip(_STRESS, m, strict=True))
        source = {}
        for name, val in comps.items():
            if val != 0.0:
                f = np.zeros((self.n, self.n, self.n))
                f[i, j, k] = float(amplitude) * float(val)
                source[name] = f
        return source

    def point_force_source(self, position, force, amplitude=1.0):
        """A point body force at ``position`` -- a velocity-rate injection ``f / rho``.

        ``force`` is the 3-vector ``(fx, fy, fz)``. A vertical force (``fz``) radiates a strong P lobe
        downward and S lobes to the sides; combined force directions excite arbitrary radiation patterns.
        Returns a ``source`` dict for :meth:`step` that adds the force to the velocity rate at the node.
        """
        i, j, k = (int(round(p)) for p in position)
        force = np.asarray(force, dtype=float).reshape(3)
        source = {}
        for name, comp in zip(_VEL, force, strict=True):
            if comp != 0.0:
                f = np.zeros((self.n, self.n, self.n))
                # a body force enters dv/dt as f/rho; scale by the local inverse density
                f[i, j, k] = float(amplitude) * float(comp) * float(self._inv_rho[i, j, k])
                source[name] = f
        return source
