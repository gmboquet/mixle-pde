"""Semi-analytical finite element (SAFE) guided-wave dispersion for an isotropic plate (Lamb + SH modes).

Ultrasonic guided waves are the workhorse of NDE / structural health monitoring: a wave launched along a
plate or pipe travels far and interrogates the whole cross-section, and the way its speed varies with
frequency (its dispersion) is the fingerprint that decodes thickness, material state, and defects. The
dispersion curves come from a boundary-value problem across the waveguide cross-section, which for a plate
has the classical but transcendental Rayleigh-Lamb form -- awkward to root-find and to differentiate.

The SAFE method turns that transcendental root search into a matrix eigenproblem. Assume harmonic
propagation along the plate, ``u(x, y, t) = U(y) exp(i (k x - omega t))``, and discretize only the
remaining one-dimensional dependence across the thickness ``y`` with 1D finite elements. Substituting into
the elastodynamic weak form leaves, at each angular frequency ``omega``, the quadratic eigenvalue problem in
the wavenumber ``k``

    (K1 + i k K2 + k^2 K3 - omega^2 M) U = 0,

with ``K2`` skew-symmetric and every matrix a thin banded assembly over the through-thickness nodes. The
plate carries two families that decouple: the in-plane motion ``(ux, uy)`` gives the Lamb modes (symmetric
S and antisymmetric A), and the out-of-plane motion ``uz`` gives the shear-horizontal (SH) modes, which
have the exact closed form ``c_n = c_s / sqrt(1 - (n pi c_s / (omega 2h))^2)`` used to validate the method.

The wavenumbers ``k(omega)`` are recovered by companion linearization of the quadratic eigenproblem into a
``2N`` generalized eigenproblem and solving it; the phase velocity is ``c = omega / k``. Because the element
matrices are assembled from the thickness and the moduli through the differentiable ``ops`` tensors and the
eigen-solve is autograd-capable, the whole map ``(thickness, c_l, c_s, rho) -> dispersion curves`` is
differentiable, so a measured curve can be inverted for plate thickness or elastic moduli by gradient
descent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["safe_dispersion", "DispersionCurves", "SAFEPlate"]


@dataclass
class DispersionCurves:
    """Guided-wave dispersion at one frequency: propagating modes sorted by phase velocity.

    ``freq`` is the frequency in Hz, ``fd`` the frequency-thickness product (Hz*m, the standard dispersion
    abscissa). ``k`` are the real wavenumbers (rad/m) of the propagating modes, ``cph = omega / k`` their
    phase velocities (m/s), and ``kind`` the mode family of each entry (``"S"`` / ``"A"`` for symmetric /
    antisymmetric Lamb, ``"SH"`` for shear-horizontal). Arrays are aligned and sorted by ascending ``cph``.
    """

    freq: float
    fd: float
    k: np.ndarray
    cph: np.ndarray
    kind: list[str]

    def phase_velocity(self, kind: str, order: int = 0) -> float:
        """The phase velocity (m/s) of the ``order``-th mode of a family (0 = fundamental).

        ``kind`` is ``"S"``, ``"A"`` or ``"SH"``; modes of that family are taken in ascending cutoff order
        (S0/A0/SH0 first). Raises ``IndexError`` if that mode is not propagating at this frequency.
        """
        sel = [c for c, m in zip(self.cph, self.kind, strict=True) if m == kind]
        sel.sort()
        return float(sel[order])


class SAFEPlate:
    """Semi-analytical finite element model of an isotropic plate for guided-wave dispersion.

    ``SAFEPlate(thickness, c_l=..., c_s=..., rho=..., n_elem=...)`` discretizes the plate thickness with
    ``n_elem`` linear (P1) 1D elements and pre-assembles the through-thickness FE matrices for both the
    in-plane (Lamb) and out-of-plane (SH) problems. Call :meth:`dispersion` at a frequency to solve the
    quadratic eigenproblem for the propagating wavenumbers, or use the module-level :func:`safe_dispersion`.

    Args:
        thickness: plate thickness (m). May be a plain float or an ``ops`` tensor for a differentiable model.
        c_l: longitudinal (P) wave speed (m/s), ``sqrt((lambda + 2 mu) / rho)``.
        c_s: shear (S) wave speed (m/s), ``sqrt(mu / rho)``.
        rho: density (kg/m^3).
        n_elem: number of through-thickness finite elements (mesh resolution).
    """

    def __init__(
        self,
        thickness: float,
        *,
        c_l: float = 6300.0,
        c_s: float = 3200.0,
        rho: float = 2700.0,
        n_elem: int = 40,
    ):
        self.thickness = thickness
        self.c_l = float(c_l)
        self.c_s = float(c_s)
        self.rho = float(rho)
        self.n_elem = int(n_elem)
        self.n_node = self.n_elem + 1
        # element connectivity (fixed integer structure; the differentiable scaling rides on the moduli/h)
        self._elems = [(e, e + 1) for e in range(self.n_elem)]
        # 2-point Gauss on the reference element [0, 1]
        g = 0.5 / np.sqrt(3.0)
        self._gauss_xi = np.array([0.5 - g, 0.5 + g])
        self._gauss_w = np.array([0.5, 0.5])

    # ---- differentiable assembly of the SAFE matrices ------------------------------------------------
    def _element_length(self, ops):
        """The (differentiable) element length ``Le = thickness / n_elem`` as an ops tensor."""
        h = self.thickness
        h_t = h if _is_tensor(h, ops) else ops.tensor(float(h))
        return h_t / self.n_elem

    def _sh_matrices(self, ops):
        """Assemble the SH SAFE matrices ``(K1, K3, M)`` with ``(K1 + k^2 K3 - omega^2 M) U = 0``.

        One DOF per node (the out-of-plane displacement ``uz``). ``K1 = mu * int N' N'^T`` (the through-
        thickness shear stiffness), ``K3 = mu * int N N^T`` (the in-plane shear stiffness that carries the
        ``k^2`` term), ``M = rho * int N N^T`` (the consistent mass). All scale with the element length, so
        gradients flow to ``thickness``.
        """
        n = self.n_node
        Le = self._element_length(ops)
        mu = self.rho * self.c_s**2
        # reference-element integrals (constant per element, scaled by Le or 1/Le)
        k_ref = ops.tensor(np.array([[1.0, -1.0], [-1.0, 1.0]])) / Le  # int N' N'^T
        m_ref = ops.tensor(np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0) * Le  # int N N^T
        K1 = ops.zeros(n, n)
        K3 = ops.zeros(n, n)
        M = ops.zeros(n, n)
        for a, b in self._elems:
            K1 = _scatter2(K1, (a, b), mu * k_ref, ops)
            K3 = _scatter2(K3, (a, b), mu * m_ref, ops)
            M = _scatter2(M, (a, b), self.rho * m_ref, ops)
        return K1, K3, M

    def _lamb_matrices(self, ops):
        """Assemble the in-plane (Lamb) SAFE matrices ``(K1, K2, K3, M)`` with 2 DOF/node ``(ux, uy)``.

        Strain ``eps = i k B1 U + B2 U`` with ``B1 = Lx N`` (the ``d/dx -> i k`` part) and ``B2 = Ly N'``
        (the through-thickness part), and isotropic plane-strain constitutive ``D``. Collecting powers of
        ``i k`` in ``int eps^T D eps`` gives ``K3 = int B1^T D B1``, ``K1 = int B2^T D B2`` and the skew
        ``K2 = int (B2^T D B1 - B1^T D B2)`` (the coefficient of ``i k`` in the Hermitian operator).
        """
        n = self.n_node
        ndof = 2 * n
        Le = self._element_length(ops)
        mu = self.rho * self.c_s**2
        lam = self.rho * self.c_l**2 - 2.0 * mu
        Dm = np.array(
            [
                [lam + 2.0 * mu, lam, 0.0],
                [lam, lam + 2.0 * mu, 0.0],
                [0.0, 0.0, mu],
            ]
        )
        Lx = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
        Ly = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        # The reference-element integrals depend on Le only through simple powers, and Le is a (differentiable)
        # tensor, so assemble the Le-independent pieces in numpy here and apply the scalar Le powers with ops
        # below. ke1_a scales as 1/Le (two through-thickness derivatives), ke3_a and me_a as Le (no derivative),
        # ke2_a as 1 (one derivative).
        ke1_a = np.zeros((4, 4))
        ke3_a = np.zeros((4, 4))
        ke2_a = np.zeros((4, 4))
        me_a = np.zeros((4, 4))
        for xi, w in zip(self._gauss_xi, self._gauss_w, strict=True):
            N0, N1 = 1.0 - xi, xi
            Nmat = np.array([[N0, 0.0, N1, 0.0], [0.0, N0, 0.0, N1]])  # 2x4
            dNmat_ref = np.array([[-1.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 1.0]])  # d/dxi (2x4)
            B1 = Lx @ Nmat  # 3x4  (coeff of i k)
            B2ref = Ly @ dNmat_ref  # 3x4  (coeff of d/dy, still in reference; divide by Le)
            ke3_a += w * (B1.T @ Dm @ B1)  # * Le
            ke1_a += w * (B2ref.T @ Dm @ B2ref)  # * (1/Le)
            ke2_a += w * (B2ref.T @ Dm @ B1 - B1.T @ Dm @ B2ref)  # * 1
            me_a += w * self.rho * (Nmat.T @ Nmat)  # * Le
        ke1_t = ops.tensor(ke1_a) / Le
        ke3_t = ops.tensor(ke3_a) * Le
        ke2_t = ops.tensor(ke2_a)
        me_t = ops.tensor(me_a) * Le
        K1 = ops.zeros(ndof, ndof)
        K2 = ops.zeros(ndof, ndof)
        K3 = ops.zeros(ndof, ndof)
        M = ops.zeros(ndof, ndof)
        for a, b in self._elems:
            dofs = (2 * a, 2 * a + 1, 2 * b, 2 * b + 1)
            K1 = _scatter4(K1, dofs, ke1_t, ops)
            K2 = _scatter4(K2, dofs, ke2_t, ops)
            K3 = _scatter4(K3, dofs, ke3_t, ops)
            M = _scatter4(M, dofs, me_t, ops)
        return K1, K2, K3, M

    # ---- eigen-solve ---------------------------------------------------------------------------------
    def dispersion(self, freq: float, ops, *, c_max: float = 15000.0) -> DispersionCurves:
        """Solve the SAFE eigenproblems at ``freq`` (Hz) and return the propagating dispersion.

        The SH problem is a linear generalized eigenproblem for ``k^2``; the Lamb problem is the quadratic
        eigenproblem linearized to size ``2N``. Only real, forward-propagating wavenumbers with phase
        velocity below ``c_max`` are kept (complex / evanescent branches are discarded). Modes are labelled
        ``"SH"`` for shear-horizontal and ``"S"`` / ``"A"`` for symmetric / antisymmetric Lamb by the parity
        of the eigenvector about the mid-plane.
        """
        t = ops._t
        omega = 2.0 * np.pi * float(freq)

        # -- SH: (omega^2 M - K1) U = k^2 K3 U -> k^2 = eig(K3^{-1} (omega^2 M - K1)) --
        K1s, K3s, Ms = self._sh_matrices(ops)
        A_sh = omega**2 * Ms - K1s
        sh_op = t.linalg.solve(K3s, A_sh)
        k2_sh = t.linalg.eigvals(sh_op)
        sh_k, sh_c = _propagating_from_k2(k2_sh, omega, c_max)

        # -- Lamb: (K1 + i k K2 + k^2 K3 - omega^2 M) U = 0, companion linearization --
        K1l, K2l, K3l, Ml = self._lamb_matrices(ops)
        A = K1l - omega**2 * Ml
        ndof = A.shape[0]
        eye = t.eye(ndof, dtype=A.dtype)
        zero = t.zeros((ndof, ndof), dtype=A.dtype)
        cA = A.to(t.complex128)
        cK2 = K2l.to(t.complex128)
        cK3 = K3l.to(t.complex128)
        cI = eye.to(t.complex128)
        cZ = zero.to(t.complex128)
        top = t.cat([cZ, cI], dim=1)
        bot = t.cat([-cA, -1j * cK2], dim=1)
        AA = t.cat([top, bot], dim=0)
        top2 = t.cat([cI, cZ], dim=1)
        bot2 = t.cat([cZ, cK3], dim=1)
        BB = t.cat([top2, bot2], dim=0)
        gen = t.linalg.solve(BB, AA)
        kvals, kvecs = t.linalg.eig(gen)
        lamb_k, lamb_c, lamb_kind = _propagating_lamb(kvals, kvecs, ndof, omega, c_max)

        k_all = np.concatenate([sh_k, lamb_k]) if len(sh_k) or len(lamb_k) else np.array([])
        c_all = np.concatenate([sh_c, lamb_c]) if len(sh_c) or len(lamb_c) else np.array([])
        kinds = ["SH"] * len(sh_k) + lamb_kind
        order = np.argsort(c_all) if len(c_all) else np.array([], dtype=int)
        fd = float(freq) * float(_as_float(self.thickness))
        return DispersionCurves(
            freq=float(freq),
            fd=fd,
            k=k_all[order],
            cph=c_all[order],
            kind=[kinds[i] for i in order],
        )


def safe_dispersion(
    freq: float,
    thickness: float,
    ops,
    *,
    c_l: float = 6300.0,
    c_s: float = 3200.0,
    rho: float = 2700.0,
    n_elem: int = 40,
    c_max: float = 15000.0,
) -> DispersionCurves:
    """SAFE guided-wave dispersion of an isotropic plate at one frequency.

    Convenience wrapper that builds a :class:`SAFEPlate` and solves its dispersion. Returns a
    :class:`DispersionCurves` holding the propagating Lamb (S/A) and SH modes sorted by phase velocity.

    Args:
        freq: frequency (Hz).
        thickness: plate thickness (m); a float or an ``ops`` tensor for a differentiable model.
        ops: the backend math namespace (``mixle_pde.ops.make_ops()``).
        c_l, c_s: longitudinal and shear wave speeds (m/s).
        rho: density (kg/m^3).
        n_elem: through-thickness element count (mesh resolution).
        c_max: discard modes with phase velocity above this (m/s).
    """
    plate = SAFEPlate(thickness, c_l=c_l, c_s=c_s, rho=rho, n_elem=n_elem)
    return plate.dispersion(freq, ops, c_max=c_max)


# ------------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------------
def _is_tensor(x, ops) -> bool:
    return ops._t.is_tensor(x)


def _as_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(x.detach().cpu().item())


def _scatter2(mat, nodes, block, ops):
    """Add a 2x2 element ``block`` at node DOFs ``nodes`` into ``mat`` (SH: one DOF per node)."""
    t = ops._t
    idx = t.tensor(list(nodes), dtype=t.long)
    r = idx.repeat_interleave(2)
    c = idx.repeat(2)
    return mat.index_put((r, c), mat[r, c] + block.reshape(-1), accumulate=False)


def _scatter4(mat, dofs, block, ops):
    """Add a 4x4 element ``block`` at global DOFs ``dofs`` into ``mat`` (Lamb: two DOF per node)."""
    t = ops._t
    idx = t.tensor(list(dofs), dtype=t.long)
    r = idx.repeat_interleave(4)
    c = idx.repeat(4)
    return mat.index_put((r, c), mat[r, c] + block.reshape(-1), accumulate=False)


def _propagating_from_k2(k2, omega, c_max):
    """Real positive wavenumbers from squared-wavenumber eigenvalues (the SH branch)."""
    k2n = k2.detach().cpu().numpy()
    mask = np.abs(k2n.imag) < 1e-6 * (np.abs(k2n).max() + 1.0)
    real_k2 = k2n.real[mask]
    real_k2 = real_k2[real_k2 > 0.0]
    k = np.sqrt(real_k2)
    c = omega / k
    keep = c <= c_max
    return k[keep], c[keep]


def _propagating_lamb(kvals, kvecs, ndof, omega, c_max):
    """Real forward wavenumbers and S/A labels from the linearized Lamb eigenproblem."""
    kv = kvals.detach().cpu().numpy()
    vv = kvecs.detach().cpu().numpy()
    scale = np.abs(kv).max() + 1.0
    ks: list[float] = []
    cs: list[float] = []
    kinds: list[str] = []
    seen: list[float] = []
    for i in range(len(kv)):
        kk = kv[i]
        if abs(kk.imag) > 1e-3 * scale:
            continue
        kr = kk.real
        if kr <= 1.0:  # forward propagating, drop the ~0 and backward roots
            continue
        c = omega / kr
        if c > c_max:
            continue
        if any(abs(kr - s) < 1e-3 * (abs(s) + 1.0) for s in seen):
            continue
        seen.append(kr)
        U = vv[:ndof, i]
        kind = _classify_lamb(U)
        if kind == "?":
            continue
        ks.append(kr)
        cs.append(c)
        kinds.append(kind)
    return np.array(ks), np.array(cs), kinds


def _classify_lamb(U):
    """Symmetric (S) vs antisymmetric (A) Lamb mode from an eigenvector's parity about the mid-plane.

    ``ux`` symmetric and ``uy`` antisymmetric is the symmetric (S) family; the reverse is antisymmetric (A).
    Parity is the correlation of the field with its through-thickness reversal.
    """
    ux = U[0::2]
    uy = U[1::2]

    def parity(a):
        ar = a[::-1]
        den = np.vdot(a, a)
        return float(np.real(np.vdot(a, ar) / den)) if abs(den) > 1e-30 else 0.0

    sux, suy = parity(ux), parity(uy)
    if sux > 0.3 and suy < -0.3:
        return "S"
    if sux < -0.3 and suy > 0.3:
        return "A"
    return "?"
