"""The digital twin of a physical site: monitoring in, what-ifs out (work-plan workstream P, P3).

A :class:`DigitalTwin` is the living calibrated forward model of one site. It pairs a static
:class:`~mixle_pde.latent.PosteriorField3D` -- the twin's current best estimate of the field, `state`
-- with everything needed to update or drive that estimate: the site's :class:`~mixle_pde.latent.Field3D`
geometry, a :class:`~mixle_pde.observations.ForwardOperatorRegistry` resolving observation kinds to
physics, and a :class:`~mixle_pde.field_inversion.FieldGaussianPrior` regularising the estimate where
data is thin.

Two operations move the twin:

* :meth:`DigitalTwin.assimilate` ingests a batch monitoring sweep (a list of
  :class:`~mixle_pde.observations.Observation` s per time) through
  :func:`mixle_pde.field_assimilation.assimilate_4d_ensemble` (the nonlinear-capable path; pass
  ``ensemble=False`` for the linear-Gaussian fast path, :func:`~mixle_pde.field_assimilation.assimilate_4d`)
  and returns a brand NEW twin whose `state` is the freshest
  :class:`~mixle_pde.field_assimilation.PosteriorField4D.at_time` slice. The twin never mutates in
  place -- every assimilation is a new, independently inspectable object, and the returned twin's grid
  provenance chains a content hash of the ingested monitoring sweep back to its parent so a receipt can
  walk "which readings moved this estimate" hop by hop.
* :meth:`DigitalTwin.forecast` seeds a :class:`~mixle_pde.simulation_service.Scenario`'s leading step
  from the twin's current calibrated mean (written as a content-hashed artifact, IC-2 style) and hands
  the rewired scenario to :func:`mixle_pde.simulation_service.simulate` -- the SAME provenanced forward
  service P1/P2 drive for any other what-if. Monitoring calibrates; `simulate` projects. `n`/`rng` draw
  ``self.state.sample(n, rng)`` to attach a lightweight posterior-spread summary (mean/std of the seeded
  field across posterior draws) to the returned :class:`~mixle_pde.simulation_service.SimResult` as
  `uncertainty` -- a snapshot of the CURRENT calibration's spread, not a Monte-Carlo forward propagation
  of that spread through the scenario's physics (full forward UQ is P4's surrogate/UQ scope, not P3's).

No online/streaming assimilation and no mechanistic evolution model beyond the random-walk / linear
dynamics :mod:`mixle_pde.field_assimilation` already supports -- batch monitoring sweeps only, per
work-plan §1.2. `forecast` composes with whatever forwards are already registered
(:func:`mixle_pde.simulation_service.register_forward`); this module registers none of its own.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from mixle_pde import simulation_service
from mixle_pde.field_assimilation import assimilate_4d, assimilate_4d_ensemble
from mixle_pde.field_inversion import FieldGaussianPrior
from mixle_pde.io.artifacts import sha256_of_arrays
from mixle_pde.latent import PROVENANCE_HASH_KEY, Field3D, PosteriorField3D
from mixle_pde.observations import ForwardOperatorRegistry
from mixle_pde.simulation_service import Scenario, ScenarioStep, SimResult

_SEED_ARRAY_KEY = "field"  # the array name `forecast` writes the calibrated mean under and re-reads.


def _monitoring_content_hash(observations_by_time, times: np.ndarray) -> str:
    """A deterministic sha256 (the IC-2 hashing rule) over one monitoring sweep's raw arrays.

    Not an IC-2 artifact (a monitoring sweep is not a saved posterior), but the same
    ``sha256_of_arrays`` rule, so a receipt walking data -> assimilation -> forecast reads one
    consistent hashing convention at every hop.
    """
    arrays: dict[str, np.ndarray] = {"times": np.asarray(times, dtype=float)}
    for t, obs_list in enumerate(observations_by_time):
        for i, obs in enumerate(obs_list):
            arrays[f"t{t:04d}_obs{i:04d}_value"] = np.asarray(obs.value, dtype=float)
            arrays[f"t{t:04d}_obs{i:04d}_location"] = np.asarray(obs.location, dtype=float)
    return sha256_of_arrays(arrays)


@dataclass
class DigitalTwin:
    """The living calibrated forward model of one site: monitoring in, what-ifs out."""

    grid: Field3D
    registry: ForwardOperatorRegistry
    prior: FieldGaussianPrior
    state: PosteriorField3D  # current calibrated field estimate
    process_var: float = 1e-2

    def assimilate(self, observations_by_time, times, *, ensemble: bool = True) -> DigitalTwin:
        """Ingest a batch monitoring sweep; return a NEW twin with the freshest calibrated `state`.

        ``observations_by_time``/``times`` are the :func:`~mixle_pde.field_assimilation.assimilate_4d_ensemble`
        shapes: one (possibly empty) list of :class:`~mixle_pde.observations.Observation` per entry of
        ``times``. ``ensemble=True`` (default) uses the nonlinear-capable
        :func:`~mixle_pde.field_assimilation.assimilate_4d_ensemble`; ``ensemble=False`` uses the
        linear-Gaussian fast path :func:`~mixle_pde.field_assimilation.assimilate_4d` (only valid when
        every registered forward operator used is linear). Either way `state` becomes the posterior
        slice at the LAST assimilated time -- the twin's freshest estimate.
        """
        times_arr = np.asarray(times, dtype=float)
        assimilate_fn = assimilate_4d_ensemble if ensemble else assimilate_4d
        posterior4d = assimilate_fn(
            self.grid,
            times_arr,
            observations_by_time,
            self.registry,
            self.prior,
            process_var=self.process_var,
        )
        latest = posterior4d.at_time(float(times_arr[-1]))

        monitoring_hash = _monitoring_content_hash(observations_by_time, times_arr)
        parent_hash = self.grid.provenance.get(PROVENANCE_HASH_KEY)
        # A fresh Field3D copy, not an in-place mutation of `self.grid` -- assimilate() never touches
        # the twin it was called on, so the old twin stays a valid, independently inspectable snapshot.
        new_grid = dataclasses.replace(self.grid, provenance=dict(self.grid.provenance))
        new_grid.attach_content_hash(monitoring_hash, stage="digital_twin_assimilate", parent=parent_hash)
        new_state = dataclasses.replace(latest, grid=new_grid)

        return DigitalTwin(
            grid=new_grid,
            registry=self.registry,
            prior=self.prior,
            state=new_state,
            process_var=self.process_var,
        )

    def forecast(self, scenario: Scenario, *, n: int = 128, rng=None) -> SimResult:
        """Seed ``scenario``'s leading step from `state.mean` and run it through `simulate`.

        The current calibrated mean is written as a content-hashed artifact (IC-2 style) and the
        scenario's first step's ``inputs_ref`` is rewired onto it before dispatch, so the forecast is
        provenance-linked to the exact twin state that produced it. The returned
        :class:`~mixle_pde.simulation_service.SimResult` additionally carries the posterior spread of
        the seeded field (mean/std over ``n`` draws of ``state.sample``) under ``uncertainty`` -- a
        snapshot of the CURRENT calibration's uncertainty, not a re-propagation of it through the
        scenario's physics.
        """
        rng = np.random.default_rng() if rng is None else rng
        store_dir = simulation_service._default_store_dir()
        seed_ref = simulation_service.write_result_artifact(
            {_SEED_ARRAY_KEY: np.asarray(self.state.mean, dtype=float)},
            grid={"shape": [self.grid.n]},
            units=self.grid.units,
            provenance={
                "stage": "digital_twin_forecast_seed",
                "twin_content_hash": self.grid.provenance.get(PROVENANCE_HASH_KEY),
            },
            store_dir=store_dir,
        )

        steps = list(scenario.steps)
        leading = steps[0]
        steps[0] = ScenarioStep(op=leading.op, inputs_ref=seed_ref, params=leading.params)
        seeded_scenario = Scenario(
            steps=steps,
            couplings=scenario.couplings,
            provenance={**scenario.provenance, "digital_twin_seed": seed_ref},
        )

        result = simulation_service.simulate(seeded_scenario)

        draws = self.state.sample(int(n), rng)
        result.uncertainty = {
            "mean": draws.mean(axis=0),
            "std": draws.std(axis=0, ddof=1) if n > 1 else np.zeros_like(draws[0]),
        }
        return result


def build_twin(
    grid: Field3D,
    registry: ForwardOperatorRegistry,
    prior: FieldGaussianPrior,
    *,
    process_var: float = 1e-2,
    rng=None,  # noqa: ARG001 -- accepted for symmetry with `forecast`; the un-updated baseline below
    # is the prior's exact closed-form mean/covariance (the same identity-transform requirement
    # `assimilate_4d`/`assimilate_4d_ensemble` impose), so no randomness is needed to build it.
) -> DigitalTwin:
    """Initialise a :class:`DigitalTwin` from `prior` alone -- the un-updated baseline model.

    ``state`` is the prior's own Gaussian: mean/MAP ``prior.mean_vector(grid)`` and the exact dense
    covariance ``prior.precision(grid)^-1``. Requires an identity-transform field (``bounds=None``),
    the same requirement :func:`~mixle_pde.field_assimilation.assimilate_4d_ensemble` imposes on the
    fields it can assimilate.
    """
    if grid.bounds is not None:
        raise ValueError("build_twin currently requires an identity-transform field (bounds=None).")
    mean = prior.mean_vector(grid)
    precision = prior.precision(grid)
    cov = np.linalg.inv(precision + 1.0e-9 * np.eye(grid.n))
    state = PosteriorField3D(grid=grid, mean=mean, map=mean.copy(), cov=cov)
    return DigitalTwin(grid=grid, registry=registry, prior=prior, state=state, process_var=process_var)
