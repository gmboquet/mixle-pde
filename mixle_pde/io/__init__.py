"""Real-world dataset ingest for :mod:`mixle_pde` (workstream B).

Each sibling module here (``segy``, ``las``, ``potfield``, ...) reads one field-survey or well-log
file format into the plain-array dataclasses the rest of the package already works with -- a
:class:`~mixle_pde.observations.Observation`, a :class:`~mixle_pde.latent.Field3D`, or a
format-specific carrier such as :class:`mixle_pde.io.las.WellLog` -- so an inversion or forward model
never has to know a file format, only an array shape. Heavy third-party readers (``segyio``,
``lasio``, ``rasterio``, ...) are imported lazily inside the function that needs them, never at
package import time, so importing :mod:`mixle_pde` never requires every optional geophysics backend
to be installed.
"""

from __future__ import annotations
