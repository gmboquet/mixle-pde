# Architecture decision records

Backend-selection ADRs record why this repo has or has not adopted a given external numerical backend, and
under what conditions that should change. The template and discipline (YAML front matter, grounded
capability-gap claims, an explicit license/packaging cost, an unhedged Adopt/Defer-until-X/Never verdict) follow
`mixle-discrete`'s `docs/adr/0000-backend-adapter-template.md` (`DISC-D0.9`, PR #5). This directory does not
duplicate that template file verbatim -- see `0001-mp-a4-backend-selection.md`'s "Format note" section for how
and why a portfolio-selection record adapts it.

- `0001-mp-a4-backend-selection.md` -- `MP-A4`: Gmsh, DOLFINx/FFCx, PETSc, MPI, preCICE, and OpenFOAM evaluated
  against `mixle_pde/pde_backend_registry.py`'s legacy FD/FDTD/FEM/spectral kernels and
  `mixle_pde/linear_solve.py`'s SciPy-based solvers.
