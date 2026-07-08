Security And Data Handling
==========================

``mixle-pde`` works with physical simulations, geophysical observations, and
field-inversion artifacts. Public docs should distinguish synthetic examples
from real subsurface or operational data.

Synthetic Versus Real Data
--------------------------

Examples and tests should say whether observations are synthetic, public, or
private. Synthetic inversions are useful validation fixtures, but they should
not be described as real exploration or engineering recommendations.

Operational Data
----------------

Real geophysical, environmental, industrial, or medical PDE data may be
sensitive. Do not commit private survey data, coordinates, well logs,
proprietary meshes, or customer payloads in tests, docs, or generated reports.

Artifacts
---------

Posterior fields, meshes, ensembles, and validation reports should carry units,
coordinate assumptions, source inputs, and limitations. Applications should
export summaries with provenance rather than raw arrays alone.

Numerical Claims
----------------

Solver examples should state whether they are analytic checks, manufactured
solutions, synthetic recovery tests, or production-scale simulations. Avoid
implying that a smoke test validates a full physical modeling workflow.

Release Checklist
-----------------

Before release:

* run focused numerical tests;
* label synthetic/public/private inputs;
* inspect generated artifacts for private paths or coordinates;
* document optional dependencies used for validation;
* build docs with warnings as errors.
