"""3D anisotropic (VTI / TTI) elastodynamic wave equation on a staggered velocity-stress grid.

The isotropic stepper in :mod:`mixle_pde.elastic` hardwires a two-constant (Lame) stiffness into the
stress-rate update ``dsigma_ij/dt = C_ijkl dv_k/dl``. Sedimentary rock is not isotropic: fine layering and
aligned cracks make the medium transversely isotropic, with a distinct wave speed along the symmetry axis
and across it. This solver replaces the two-constant law with a general hexagonal (VTI) stiffness carrying
the five independent constants

    c11, c33, c44, c66, c13     (Voigt notation, symmetry axis along z),

or, equivalently, the Thomsen parameters ``(Vp0, Vs0, epsilon, delta, gamma)`` that geophysicists actually
measure. A ``tilt`` angle rotates the symmetry axis away from vertical through a Bond transform of the full
21-component stiffness, giving a tilted-transverse-isotropic (TTI) medium such as a dipping shale.

The rest of the machinery is inherited from the isotropic template: the same Virieux (1986) velocity-stress
staggered grid (three velocities and six stresses a half step apart), the same forward/backward one-sided
differences that centre every derivative at its update location, the same absorbing sponge, moment-tensor
and point-force sources, and full differentiability through ``ops`` (so a seismic observation is a
:class:`~mixle_pde.inverse.Differential` forward and the gradient w.r.t. the anisotropy parameters is the
anisotropic full-waveform-inversion sensitivity).

VTI Voigt stiffness (symmetry axis z):

    | c11 c12 c13  0   0   0  |
    | c12 c11 c13  0   0   0  |          c12 = c11 - 2 c66
    | c13 c13 c33  0   0   0  |          c22 = c11,  c55 = c44,  c23 = c13
    |  0   0   0  c44  0   0  |
    |  0   0   0   0  c44  0  |
    |  0   0   0   0   0  c66 |

Thomsen mapping (Thomsen, 1986):

    c33 = rho Vp0^2,  c44 = rho Vs0^2
    c11 = c33 (1 + 2 epsilon)
    c66 = c44 (1 + 2 gamma)
    c13 = sqrt( 2 c33 (c33 - c44) delta + (c33 - c44)^2 ) - c44

Analytical phase velocities along the symmetry axes (the acceptance benchmark):

    vertical   P  = sqrt(c33/rho) = Vp0
    horizontal P  = sqrt(c11/rho) = Vp0 sqrt(1 + 2 epsilon)
    vertical   S  = sqrt(c44/rho) = Vs0
    horizontal SH = sqrt(c66/rho) = Vs0 sqrt(1 + 2 gamma)
"""

from __future__ import annotations

import numpy as np

# component order in the packed state: three velocities then six stresses (symmetric tensor)
_VEL = ("vx", "vy", "vz")
_STRESS = ("sxx", "syy", "szz", "sxy", "sxz", "syz")
_COMPONENTS = _VEL + _STRESS

# Voigt index of each engineering strain rate produced by the velocity gradients, in the order
# (exx, eyy, ezz, 2 eyz, 2 exz, 2 exy) = Voigt rows (0..5). We keep this canonical Voigt order and map to
# the staggered stress components at assembly time.
_VOIGT = ("xx", "yy", "zz", "yz", "xz", "xy")


def thomsen_to_cij(vp0, vs0, epsilon, delta, gamma, rho):
    """Map Thomsen parameters ``(Vp0, Vs0, epsilon, delta, gamma)`` + density to the 5 VTI constants.

    Returns ``(c11, c33, c44, c66, c13)``. Scalars or broadcastable arrays; the ``delta`` branch uses the
    exact (anelliptic) form ``c13 = sqrt(2 c33 (c33 - c44) delta + (c33 - c44)^2) - c44``.
    """
    vp0 = np.asarray(vp0, dtype=float)
    vs0 = np.asarray(vs0, dtype=float)
    epsilon = np.asarray(epsilon, dtype=float)
    delta = np.asarray(delta, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    rho = np.asarray(rho, dtype=float)
    c33 = rho * vp0**2
    c44 = rho * vs0**2
    c11 = c33 * (1.0 + 2.0 * epsilon)
    c66 = c44 * (1.0 + 2.0 * gamma)
    inside = 2.0 * c33 * (c33 - c44) * delta + (c33 - c44) ** 2
    c13 = np.sqrt(np.maximum(inside, 0.0)) - c44
    return c11, c33, c44, c66, c13


def vti_voigt_matrix(c11, c33, c44, c66, c13):
    """Assemble the 6x6 Voigt stiffness of a VTI medium (symmetry axis z) from the 5 constants.

    Accepts scalars (returns a plain ``(6, 6)``) or matching arrays (returns ``(..., 6, 6)``).
    """
    c11 = np.asarray(c11, dtype=float)
    shape = np.broadcast(c11, c33, c44, c66, c13).shape
    C = np.zeros(shape + (6, 6))
    c11, c33, c44, c66, c13 = np.broadcast_arrays(c11, c33, c44, c66, c13)
    c12 = c11 - 2.0 * c66
    C[..., 0, 0] = c11
    C[..., 1, 1] = c11
    C[..., 2, 2] = c33
    C[..., 0, 1] = c12
    C[..., 1, 0] = c12
    C[..., 0, 2] = c13
    C[..., 2, 0] = c13
    C[..., 1, 2] = c13
    C[..., 2, 1] = c13
    C[..., 3, 3] = c44
    C[..., 4, 4] = c44
    C[..., 5, 5] = c66
    return C


def bond_matrix(theta, axis=1):
    """The 6x6 Bond matrix ``M`` that rotates a Voigt stiffness by ``theta`` about a coordinate axis.

    For a rotation ``R`` of the physical frame, the Voigt stiffness transforms as ``C' = M C M^T``. ``axis``
    selects the rotation axis (0=x, 1=y, 2=z); a TTI tilt of the z symmetry axis toward x is a rotation
    about y (``axis=1``). ``theta`` is in radians.
    """
    c, s = np.cos(theta), np.sin(theta)
    if axis == 0:  # rotation about x
        R = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    elif axis == 1:  # rotation about y (tilts z toward x)
        R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    else:  # rotation about z
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    # Bond matrix from the rotation R (Auld, 1973). Rows/cols in Voigt order (xx,yy,zz,yz,xz,xy).
    M = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            M[i, j] = R[i, j] ** 2
    # off-diagonal upper-right (normal-from-shear) block
    pairs = [(1, 2), (0, 2), (0, 1)]  # (yz, xz, xy) index pairs
    for i in range(3):
        a, b = pairs[i]
        for j in range(3):
            M[i, 3 + j] = 2.0 * R[i, pairs[j][0]] * R[i, pairs[j][1]]
            M[3 + i, j] = R[pairs[i][0], j] * R[pairs[i][1], j]
    for i in range(3):
        ai, bi = pairs[i]
        for j in range(3):
            aj, bj = pairs[j]
            M[3 + i, 3 + j] = R[ai, aj] * R[bi, bj] + R[ai, bj] * R[bi, aj]
    return M


def rotate_voigt(C, theta, axis=1):
    """Rotate a Voigt stiffness ``C`` (``(6,6)`` or ``(...,6,6)``) by ``theta`` about ``axis``: ``M C M^T``."""
    M = bond_matrix(theta, axis)
    return M @ C @ M.T


class AnisotropicElasticWave3D:
    """A differentiable 3D anisotropic (VTI/TTI) elastic-wave stepper (velocity-stress staggered leapfrog).

    ``AnisotropicElasticWave3D(n, dt=..., spacing=..., ...)`` builds the forward on an ``n x n x n`` grid.
    Give the medium either through the 5 VTI constants ``(c11, c33, c44, c66, c13)`` or through Thomsen
    parameters ``(vp0, vs0, epsilon, delta, gamma)`` with ``rho``; a ``tilt`` angle (radians, about the y
    axis by default) rotates the symmetry axis to make a TTI medium. Each parameter may be a scalar or a
    full ``n^3`` field (a heterogeneous anisotropic model). ``tilt`` may likewise be a scalar or field.

    The state packs the nine components in the order ``(vx, vy, vz, sxx, syy, szz, sxy, sxz, syz)``
    (``pack``/``unpack``); ``step(state, ops, source=...)`` advances one full leapfrog step (stress a half
    step, then velocity a half step). The stress-rate law is the full Hooke tensor ``dsigma = C : grad v``
    in Voigt form, so along the symmetry axes the P/S/SH speeds are the exact Christoffel/Thomsen values.

    Staggering follows the isotropic template exactly: diagonal stresses at the node ``(i,j,k)``,
    ``vx@(i+1/2,j,k)``, ``vy@(i,j+1/2,k)``, ``vz@(i,j,k+1/2)``, and the shear stresses on the cell edges.
    The isotropic P-speed Courant bound is replaced by the fastest anisotropic P speed
    ``vp_max = sqrt(max(c11,c33)/min(rho))``; keep ``dt <= spacing / (vp_max * sqrt(3))``.
    """

    def __init__(
        self,
        n: int,
        *,
        dt: float,
        spacing: float | None = None,
        c11: float | np.ndarray | None = None,
        c33: float | np.ndarray | None = None,
        c44: float | np.ndarray | None = None,
        c66: float | np.ndarray | None = None,
        c13: float | np.ndarray | None = None,
        vp0: float | np.ndarray | None = None,
        vs0: float | np.ndarray | None = None,
        epsilon: float | np.ndarray = 0.0,
        delta: float | np.ndarray = 0.0,
        gamma: float | np.ndarray = 0.0,
        rho: float | np.ndarray = 1.0,
        tilt: float | np.ndarray = 0.0,
        tilt_axis: int = 1,
        absorb_width: int = 0,
        absorb_strength: float = 2.0,
    ):
        self.n = int(n)
        self.dt = float(dt)
        self.h = float(spacing) if spacing is not None else 1.0 / (n - 1)
        self._nc = self.n**3  # per-component length
        self.tilt_axis = int(tilt_axis)

        rho = self._as_field(rho)
        if c11 is not None and c33 is not None and c44 is not None and c66 is not None and c13 is not None:
            c11 = self._as_field(c11)
            c33 = self._as_field(c33)
            c44 = self._as_field(c44)
            c66 = self._as_field(c66)
            c13 = self._as_field(c13)
        elif vp0 is not None and vs0 is not None:
            c11, c33, c44, c66, c13 = thomsen_to_cij(
                self._as_field(vp0),
                self._as_field(vs0),
                self._as_field(epsilon),
                self._as_field(delta),
                self._as_field(gamma),
                rho,
            )
        else:
            raise ValueError("give either the 5 VTI constants (c11,c33,c44,c66,c13) or Thomsen (vp0,vs0,...)")
        self.rho = rho
        self._inv_rho = 1.0 / rho
        # VTI Voigt matrix per cell (n^3, 6, 6), optionally Bond-rotated to a tilted symmetry axis.
        self.tilt = self._as_field(tilt)
        C = self._assemble_field_voigt(c11, c33, c44, c66, c13, self.tilt)
        self._C = C  # (n, n, n, 6, 6)
        self.vp_max = float(np.sqrt(np.max(np.maximum(c11, c33)) / np.min(rho)))
        self._gamma = self._build_sponge(absorb_width, absorb_strength)

    # ---- construction helpers ------------------------------------------------------------------------
    def _as_field(self, x):
        """Broadcast a scalar or array parameter to an ``(n, n, n)`` field."""
        a = np.asarray(x, dtype=float)
        if a.ndim == 0:
            return np.full((self.n, self.n, self.n), float(a))
        return a.reshape(self.n, self.n, self.n).astype(float)

    def _assemble_field_voigt(self, c11, c33, c44, c66, c13, tilt):
        """Build the per-cell ``(n,n,n,6,6)`` Voigt stiffness, Bond-rotating each cell by its tilt angle.

        The unrotated VTI matrix is vectorized over all cells; the rotation is applied per unique tilt
        (usually one) for speed, so a uniform tilt costs a single Bond transform.
        """
        n = self.n
        C = vti_voigt_matrix(c11, c33, c44, c66, c13)  # (n,n,n,6,6)
        tilt = np.asarray(tilt, dtype=float)
        if np.allclose(tilt, 0.0):
            return C
        flat_C = C.reshape(-1, 6, 6)
        flat_t = tilt.reshape(-1)
        out = np.empty_like(flat_C)
        # group identical tilt angles so each Bond matrix is formed once
        uniq = np.unique(np.round(flat_t, 12))
        for th in uniq:
            mask = np.round(flat_t, 12) == th
            M = bond_matrix(float(th), self.tilt_axis)
            out[mask] = M @ flat_C[mask] @ M.T
        return out.reshape(n, n, n, 6, 6)

    def _build_sponge(self, width, strength):
        """A velocity-damping ramp near the six faces (the isotropic template's sponge)."""
        n = self.n
        gamma = np.zeros((n, n, n))
        if width > 0:
            idx = np.arange(n)
            d = np.minimum(idx, n - 1 - idx)  # distance (in nodes) to the nearest edge
            ramp = np.where(d < width, (1.0 - d / width) ** 2, 0.0)
            gx = np.broadcast_to(ramp[:, None, None], (n, n, n))
            gy = np.broadcast_to(ramp[None, :, None], (n, n, n))
            gz = np.broadcast_to(ramp[None, None, :], (n, n, n))
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
    @staticmethod
    def _dp(f, axis, h):
        """Forward difference ``(f[i+1] - f[i]) / h`` centred a half cell forward of ``f`` along ``axis``."""
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        out = f * 0.0
        out[tuple(lo)] = (f[tuple(hi)] - f[tuple(lo)]) / h
        return out

    @staticmethod
    def _dm(f, axis, h):
        """Backward difference ``(f[i] - f[i-1]) / h`` centred a half cell backward of ``f`` along ``axis``."""
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(1, None)
        hi[axis] = slice(0, -1)
        out = f * 0.0
        out[tuple(lo)] = (f[tuple(lo)] - f[tuple(hi)]) / h
        return out

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

    # ---- leapfrog step -------------------------------------------------------------------------------
    def step(self, state, ops, *, source=None):
        """Advance one full velocity-stress leapfrog step (update stress, then velocity).

        The stress rate is the full anisotropic Hooke law ``dsigma_I/dt = sum_J C_IJ e_J`` in Voigt form,
        where the six engineering strain rates ``e = (dvx/dx, dvy/dy, dvz/dz, dvy/dz+dvz/dy, dvx/dz+dvz/dx,
        dvx/dy+dvy/dx)`` are the same staggered velocity gradients the isotropic template uses. ``source`` is
        an optional dict of per-component additive rate injections keyed by component name.
        """
        n, h, dt = self.n, self.h, self.dt
        vx, vy, vz, sxx, syy, szz, sxy, sxz, syz = (c.reshape(n, n, n) for c in self.unpack(state))
        C = ops.tensor(self._C)  # (n,n,n,6,6)
        inv_rho = ops.tensor(self._inv_rho)
        src = source or {}

        def add(name, field):
            s = src.get(name)
            return field if s is None else field + dt * ops.tensor(np.asarray(s).reshape(n, n, n))

        # --- strain rates centred at the stress locations ----------------------------------------------
        # normal strain rates (back-diff of the dual velocity) live at the node with sxx,syy,szz
        exx = self._dm(vx, 0, h)
        eyy = self._dm(vy, 1, h)
        ezz = self._dm(vz, 2, h)
        # shear (engineering) strain rates (fwd-diff of the dual velocity) live at the cell edges
        gyz = self._dp(vy, 2, h) + self._dp(vz, 1, h)  # 2 eyz -> syz edge (i,j+1/2,k+1/2)
        gxz = self._dp(vx, 2, h) + self._dp(vz, 0, h)  # 2 exz -> sxz edge (i+1/2,j,k+1/2)
        gxy = self._dp(vx, 1, h) + self._dp(vy, 0, h)  # 2 exy -> sxy edge (i+1/2,j+1/2,k)

        # --- stress update: sigma_I += dt sum_J C_IJ e_J ----------------------------------------------
        # Normal stresses use C rows 0..2 columns 0..2 (the normal-normal block) at the node.
        def cc(i, j):
            return C[..., i, j]

        sxx = add("sxx", sxx + dt * (cc(0, 0) * exx + cc(0, 1) * eyy + cc(0, 2) * ezz))
        syy = add("syy", syy + dt * (cc(1, 0) * exx + cc(1, 1) * eyy + cc(1, 2) * ezz))
        szz = add("szz", szz + dt * (cc(2, 0) * exx + cc(2, 1) * eyy + cc(2, 2) * ezz))

        # Shear stresses use the shear-shear diagonal (rows/cols 3..5). Average the stiffness onto the edge
        # where each shear stress lives (matching the isotropic template's mu edge-average). For VTI/TTI the
        # relevant modulus is c55(=xz), c44(=yz), c66(=xy); we read them off the rotated Voigt matrix.
        c_yz = 0.5 * (cc(3, 3) + self._roll_avg(cc(3, 3), (1, 2)))  # syz at (i,j+1/2,k+1/2)
        c_xz = 0.5 * (cc(4, 4) + self._roll_avg(cc(4, 4), (0, 2)))  # sxz at (i+1/2,j,k+1/2)
        c_xy = 0.5 * (cc(5, 5) + self._roll_avg(cc(5, 5), (0, 1)))  # sxy at (i+1/2,j+1/2,k)
        syz = add("syz", syz + dt * c_yz * gyz)
        sxz = add("sxz", sxz + dt * c_xz * gxz)
        sxy = add("sxy", sxy + dt * c_xy * gxy)

        # --- velocity update: v += dt (1/rho) div sigma -----------------------------------------------
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

    # ---- point sources -------------------------------------------------------------------------------
    def moment_tensor_source(self, position, moment, amplitude=1.0):
        """A moment-tensor point source at ``position`` -- a stress-rate injection that excites P and S.

        ``moment`` is the symmetric seismic moment tensor ``M`` (a 3x3 array or the six independent
        components ``[Mxx, Myy, Mzz, Mxy, Mxz, Myz]``). Returns a ``source`` dict for :meth:`step`.
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

        ``force`` is the 3-vector ``(fx, fy, fz)``. Returns a ``source`` dict for :meth:`step`.
        """
        i, j, k = (int(round(p)) for p in position)
        force = np.asarray(force, dtype=float).reshape(3)
        source = {}
        for name, comp in zip(_VEL, force, strict=True):
            if comp != 0.0:
                f = np.zeros((self.n, self.n, self.n))
                f[i, j, k] = float(amplitude) * float(comp) * float(self._inv_rho[i, j, k])
                source[name] = f
        return source
