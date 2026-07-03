"""Full-tensor gravity gradiometry (FTG) and vector/gradient magnetic sensitivity kernels.

:mod:`mixle_pde.geophysics` gives only the two scalar potential-field kernels that a legacy survey measures:
the vertical gravity anomaly ``g_z`` (:func:`~mixle_pde.geophysics.gravity_point_sensitivity`) and the
magnetic total-field anomaly ``TMI`` (:func:`~mixle_pde.geophysics.magnetic_dipole_sensitivity`). Modern
airborne/marine instruments measure much more: a full-tensor gradiometer records the five independent
second derivatives of the gravity potential, and a fluxgate/SQUID vector magnetometer records the three
field components (and their gradients). Each extra channel is another linear row in ``d = G @ model``, so it
sharpens a severely ill-posed inversion at no change to the machinery -- these operators feed
:func:`mixle_pde.geophysics.regularized_gauss_newton` and ``joint_inversion`` unchanged.

Everything here is the *point-source* kernel (each cell a point mass / point dipole at its centre), the same
approximation as the scalar kernels it extends, and everything is a plain dense NumPy sensitivity matrix
linear in the model:

* :func:`gravity_gradient_tensor` -- the five independent gravity-gradient components
  ``T_ij = d2U/dx_i dx_j`` from point masses, stacked into one ``G``. ``T_zz = -(T_xx + T_yy)`` by Laplace,
  so it is dropped from the independent set (and recoverable via :func:`trace_free_zz`).
* :func:`magnetic_vector_sensitivity` -- the three magnetic field components (or a chosen subset) from point
  dipoles under the induced-magnetization approximation, stacked into one ``G``.
* :func:`magnetic_gradient_tensor` -- the magnetic gradient tensor ``dB_i/dx_j`` (the magnetic-gradiometry
  channels), stacked into one ``G``.

The closed forms are exact for point sources: the gravity gradient of a point mass is
``T_ij = G_grav m (3 r_i r_j - |r|^2 delta_ij) / |r|^5`` (``r = obs - source``), which is manifestly
trace-free off the source (``sum_i T_ii = G_grav m (3|r|^2 - 3|r|^2)/|r|^5 = 0``).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "gravity_gradient_tensor",
    "gravity_gradient_components",
    "trace_free_zz",
    "magnetic_vector_sensitivity",
    "magnetic_gradient_tensor",
]

G_GRAV = 6.674e-11  # gravitational constant, m^3 kg^-1 s^-2 (matches geophysics.gravity_point_sensitivity)

# Eotvos: gravity gradients are quoted in Eotvos units, 1 E = 1e-9 s^-2. 1 (SI, s^-2) = 1e9 E.
EOTVOS = 1.0e9

# The five independent gravity-gradient components (Tzz = -(Txx + Tyy) by Laplace).
_FTG_COMPONENTS = ("xx", "xy", "xz", "yy", "yz")
_AXIS = {"x": 0, "y": 1, "z": 2}


def _displacements(obs, cells):
    """``d = obs - cell`` (n_obs, n_cells, 3) and its regularized norm (n_obs, n_cells)."""
    obs = np.asarray(obs, float)
    cells = np.asarray(cells, float)
    d = obs[:, None, :] - cells[None, :, :]
    r = np.maximum(np.linalg.norm(d, axis=2), 1e-6)
    return d, r


def _gg_component(d, r, i, j):
    """Per-cell gravity-gradient factor ``(3 r_i r_j - |r|^2 delta_ij) / |r|^5``.

    Multiplying by ``G_grav * V`` gives ``T_ij`` (SI, s^-2) for a unit-density point mass of volume ``V``.
    """
    ri, rj = d[:, :, i], d[:, :, j]
    delta = 1.0 if i == j else 0.0
    return (3.0 * ri * rj - delta * r**2) / r**5


def gravity_gradient_components(obs, cells, volumes, *, components=_FTG_COMPONENTS, units="eotvos"):
    r"""Per-component gravity-gradient sensitivity matrices to cell density contrast (point-mass kernel).

    For a cell of volume ``V`` at displacement ``r = obs - cell`` (metres, up-positive frame), the gravity
    gradient tensor of the point mass is the exact closed form

    .. math:: T_{ij} = G\, m\,\frac{3 r_i r_j - |r|^2\,\delta_{ij}}{|r|^5}, \qquad m = \rho\,V,

    linear in the density contrast ``rho``. This is the gradient of the *attraction* ``g = -grad U`` (i.e.
    ``T_ij = d g_i / d x_j = -d2U/dx_i dx_j``), so ``T_zz`` is minus the ``z``-derivative of the +dU/dz-signed
    :func:`~mixle_pde.geophysics.gravity_point_sensitivity` kernel. This returns one ``(n_obs, n_cells)``
    matrix per requested component, each such that ``G_ij @ rho`` is that gradient component.
    ``T_zz = -(T_xx + T_yy)`` by Laplace off the source, so only the five ``xx, xy, xz, yy, yz`` are independent.

    Args:
        obs: (n_obs, 3) observation coordinates (east, north, up), metres.
        cells: (n_cells, 3) cell-centre coordinates, metres.
        volumes: (n_cells,) cell volumes, m^3 (scalar broadcast allowed).
        components: iterable of two-character component names drawn from ``{xx, xy, xz, yy, yz, zz, yx, ...}``.
        units: ``"eotvos"`` (default; 1 E = 1e-9 s^-2, the survey unit) or ``"si"`` (s^-2 per kg/m^3).

    Returns:
        dict mapping each requested component name to its ``(n_obs, n_cells)`` sensitivity matrix.
    """
    d, r = _displacements(obs, cells)
    n_cells = d.shape[1]
    V = np.broadcast_to(np.asarray(volumes, float), (n_cells,))
    scale = G_GRAV * (EOTVOS if units == "eotvos" else 1.0)
    if units not in ("eotvos", "si"):
        raise ValueError(f"unknown units {units!r}; use 'eotvos' or 'si'.")
    out = {}
    for comp in components:
        c = comp.lower()
        if len(c) != 2 or c[0] not in _AXIS or c[1] not in _AXIS:
            raise ValueError(f"bad component {comp!r}; use two axis letters like 'xx' or 'xz'.")
        i, j = _AXIS[c[0]], _AXIS[c[1]]
        out[comp] = scale * V[None, :] * _gg_component(d, r, i, j)
    return out


def gravity_gradient_tensor(obs, cells, volumes, *, components=_FTG_COMPONENTS, units="eotvos"):
    r"""Stacked full-tensor-gradiometry sensitivity ``G`` (``n_obs * n_comp`` x ``n_cells``) to density contrast.

    Vertically stacks the requested :func:`gravity_gradient_components` (default the five independent
    ``xx, xy, xz, yy, yz``) into a single linear operator, so ``d = G @ rho`` is all component readings at all
    stations. It plugs into :func:`mixle_pde.geophysics.regularized_gauss_newton` /
    ``joint_inversion`` exactly like the scalar :func:`~mixle_pde.geophysics.gravity_point_sensitivity`.

    The block order is component-major: the first ``n_obs`` rows are the first component at every station, the
    next ``n_obs`` rows the second component, and so on (the same order as ``components``).

    Args:
        obs: (n_obs, 3) observation coordinates (east, north, up), metres.
        cells: (n_cells, 3) cell-centre coordinates, metres.
        volumes: (n_cells,) cell volumes, m^3 (scalar broadcast allowed).
        components: which gradient components to stack (default the five independent ones).
        units: ``"eotvos"`` (default) or ``"si"``.

    Returns:
        ``G`` of shape ``(n_obs * len(components), n_cells)``.
    """
    blocks = gravity_gradient_components(obs, cells, volumes, components=components, units=units)
    return np.vstack([blocks[c] for c in components])


def trace_free_zz(Gxx, Gyy):
    """Recover the ``T_zz`` sensitivity block from ``T_xx`` and ``T_yy`` via Laplace: ``T_zz = -(T_xx + T_yy)``.

    Off the source the gravity potential is harmonic, so the tensor is trace-free and ``T_zz`` carries no
    independent information. Use this when a downstream consumer wants the ``zz`` row explicitly.
    """
    return -(np.asarray(Gxx, float) + np.asarray(Gyy, float))


def _field_direction(inclination_deg, declination_deg):
    """Unit vector of the geomagnetic field in an (east, north, up) frame (inclination positive down)."""
    inc, dec = np.radians(inclination_deg), np.radians(declination_deg)
    return np.array([np.cos(inc) * np.sin(dec), np.cos(inc) * np.cos(dec), -np.sin(inc)])


def magnetic_vector_sensitivity(
    obs, cells, volumes, *, inclination, declination, field_nt=50000.0, components=("x", "y", "z")
):
    r"""Linear sensitivity of the magnetic **vector field components** to cell susceptibility (point-dipole kernel).

    Each cell is an induced point dipole of moment ``m = (T0 V / (4 pi mu0-equivalent)) kappa b`` aligned with
    the ambient field unit vector ``b``; here the constants are folded so the output is in nT and consistent
    with :func:`~mixle_pde.geophysics.magnetic_dipole_sensitivity` (whose TMI kernel is exactly the projection
    of this vector field onto ``b``). The anomalous field of a dipole at displacement ``r = obs - cell`` is

    .. math:: \mathbf{B} = \frac{T_0 V}{4\pi}\,\frac{3(\mathbf{b}\cdot\hat r)\hat r - \mathbf{b}}{|r|^3}\,\kappa,

    linear in ``kappa``. This returns one ``(n_obs, n_cells)`` matrix per requested Cartesian component, each
    such that ``G_c @ kappa`` is that field component (nT).

    Args:
        obs: (n_obs, 3) observation coordinates (east, north, up), metres.
        cells: (n_cells, 3) cell-centre coordinates, metres.
        volumes: (n_cells,) cell volumes, m^3 (scalar broadcast allowed).
        inclination, declination: geomagnetic field inclination/declination, degrees.
        field_nt: ambient field strength T0, nT.
        components: subset of ``("x", "y", "z")`` to return (default all three).

    Returns:
        dict mapping each requested component ("x"/"y"/"z") to its ``(n_obs, n_cells)`` sensitivity matrix.
    """
    b = _field_direction(inclination, declination)
    d, r = _displacements(obs, cells)
    n_cells = d.shape[1]
    V = np.broadcast_to(np.asarray(volumes, float), (n_cells,))
    rhat = d / r[:, :, None]  # (n_obs, n_cells, 3)
    bdotr = rhat @ b  # (n_obs, n_cells)
    coeff = (field_nt / (4.0 * np.pi)) * V[None, :] / r**3  # (n_obs, n_cells)
    out = {}
    for comp in components:
        c = comp.lower()
        if c not in _AXIS:
            raise ValueError(f"bad component {comp!r}; use 'x', 'y', or 'z'.")
        k = _AXIS[c]
        out[comp] = coeff * (3.0 * bdotr * rhat[:, :, k] - b[k])
    return out


def magnetic_gradient_tensor(
    obs, cells, volumes, *, inclination, declination, field_nt=50000.0, components=("xx", "xy", "xz", "yy", "yz")
):
    r"""Linear sensitivity of the magnetic **gradient tensor** ``dB_i/dx_j`` to cell susceptibility (point dipole).

    The magnetic gradiometry channels: the spatial derivative of each dipole field component. For a dipole
    moment along ``b`` at displacement ``r = obs - cell``, the field is
    ``B_i = C (3 (b.r) r_i - b_i |r|^2) / |r|^5`` with ``C = T0 V kappa / (4 pi)``, and differentiating with
    respect to the observation coordinate ``x_j`` gives the closed form

    .. math:: \frac{\partial B_i}{\partial x_j} = \frac{C}{|r|^7}
        \Big[ |r|^2\big(3 b_j r_i + 3 (\mathbf b\cdot r)\delta_{ij} - 2 b_i r_j\big)
              - 5 r_j\big(3(\mathbf b\cdot r) r_i - b_i |r|^2\big) \Big],

    linear in ``kappa``. Like the gravity tensor this magnetic tensor is symmetric and trace-free off the
    source. Returns one ``(n_obs, n_cells)`` matrix per requested ``ij`` component.

    Args:
        obs: (n_obs, 3) observation coordinates (east, north, up), metres.
        cells: (n_cells, 3) cell-centre coordinates, metres.
        volumes: (n_cells,) cell volumes, m^3 (scalar broadcast allowed).
        inclination, declination: geomagnetic field inclination/declination, degrees.
        field_nt: ambient field strength T0, nT.
        components: two-letter component names; default the five independent ones.

    Returns:
        dict mapping each requested component to its ``(n_obs, n_cells)`` sensitivity matrix (nT / m).
    """
    b = _field_direction(inclination, declination)
    d, r = _displacements(obs, cells)
    n_cells = d.shape[1]
    V = np.broadcast_to(np.asarray(volumes, float), (n_cells,))
    coeff = (field_nt / (4.0 * np.pi)) * V[None, :]  # C without the kappa
    bdotr = d @ b  # (n_obs, n_cells)
    r2 = r**2
    out = {}
    for comp in components:
        c = comp.lower()
        if len(c) != 2 or c[0] not in _AXIS or c[1] not in _AXIS:
            raise ValueError(f"bad component {comp!r}; use two axis letters like 'xy'.")
        i, j = _AXIS[c[0]], _AXIS[c[1]]
        ri, rj = d[:, :, i], d[:, :, j]
        delta = 1.0 if i == j else 0.0
        bi, bj = b[i], b[j]
        numer = r2 * (3.0 * bj * ri + 3.0 * bdotr * delta - 2.0 * bi * rj) - 5.0 * rj * (3.0 * bdotr * ri - bi * r2)
        out[comp] = coeff * numer / r**7
    return out
