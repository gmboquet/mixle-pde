"""Frequency-domain reductions for a loaded potential-field grid (workstream B4).

A raw aeromagnetic/gravity grid is not directly interpretable: a magnetic anomaly measured away
from the pole is skewed (its peak sits off to one side of the true source, and it grows a
negative side-lobe) because of the field's inclination/declination, and a survey flown at some
height above the source mixes in altitude-dependent attenuation. :func:`reduce_to_pole` and
:func:`upward_continue` are the two standard FFT filters (Blakely 1995) that undo those effects
so a grid becomes something an interpreter -- or a later inversion (workstream C) -- can read at
face value: RTP centres a magnetic anomaly over its source regardless of survey latitude, and
upward continuation projects a grid to a new, higher observation height (attenuating
high-wavenumber/near-surface noise).

Both filters are supplied by :mod:`harmonica` acting on an :class:`xarray.DataArray`; this module
only handles the plain-NumPy <-> ``DataArray`` plumbing for a :class:`~mixle_pde.io.potfield.PotentialFieldGrid`
and lazily imports ``harmonica`` (and, transitively, ``xarray``) so neither is required just to
import this module.
"""

from __future__ import annotations

import numpy as np

from mixle_pde.io.potfield import PotentialFieldGrid


def _require_harmonica():
    """Import harmonica or raise a clear ImportError naming the pip extra to install."""
    try:
        import harmonica
    except ImportError as e:
        raise ImportError(
            "potential-field reductions need the optional dependency 'harmonica'. "
            "Install it with: pip install 'mixle-pde[potfield]'  (or: pip install harmonica)."
        ) from e
    return harmonica


def _as_data_array(grid: PotentialFieldGrid):
    """Wrap a `PotentialFieldGrid` as the (northing, easting)-indexed `xarray.DataArray` harmonica expects."""
    import xarray as xr

    return xr.DataArray(
        np.asarray(grid.values, dtype=float),
        coords={"northing": np.asarray(grid.y, dtype=float), "easting": np.asarray(grid.x, dtype=float)},
        dims=("northing", "easting"),
    )


def reduce_to_pole(grid: PotentialFieldGrid, *, inclination: float, declination: float) -> np.ndarray:
    """Reduce a magnetic anomaly grid to the pole (frequency-domain RTP filter, no remanence).

    Args:
        grid: the loaded magnetic grid; assumes induced-only magnetization (the anomaly's
            magnetization direction equals the ambient field's ``inclination``/``declination``).
        inclination: inclination of the inducing geomagnetic field, degrees.
        declination: declination of the inducing geomagnetic field, degrees.

    Returns:
        The reduced-to-pole grid, same ``(ny, nx)`` shape as ``grid.values``.
    """
    harmonica = _require_harmonica()
    da = _as_data_array(grid)
    rtp = harmonica.reduction_to_pole(da, inclination=inclination, declination=declination)
    return np.asarray(rtp.values)


def upward_continue(grid: PotentialFieldGrid, height: float) -> np.ndarray:
    """Upward-continue a potential-field grid by `height` (frequency-domain FFT filter).

    Args:
        grid: the loaded gravity/magnetic grid.
        height: the (positive) height displacement to continue upward, in the same units as
            ``grid.x``/``grid.y`` (typically metres).

    Returns:
        The upward-continued grid, same ``(ny, nx)`` shape as ``grid.values``.
    """
    harmonica = _require_harmonica()
    da = _as_data_array(grid)
    continued = harmonica.upward_continuation(da, height)
    return np.asarray(continued.values)
