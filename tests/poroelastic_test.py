"""Biot poroelastic 1D solver: fast-P velocity vs Biot-Gassmann, slow-P existence, differentiability."""

from __future__ import annotations

import unittest

import numpy as np

from mixle_pde.ops import make_ops
from mixle_pde.poroelastic import (
    BiotPoroelastic1D,
    biot_gassmann_velocity,
    gassmann_moduli,
)

# a water-saturated sandstone (standard textbook values, SI units)
ROCK = dict(
    k_solid=36.0e9,
    k_fluid=2.2e9,
    k_dry=9.0e9,
    mu=7.0e9,
    phi=0.25,
    rho_solid=2650.0,
    rho_fluid=1000.0,
    eta=1.0e-3,
    permeability=1.0e-12,
    tortuosity=2.0,
)


def _ricker(t, f0, t0):
    a = (np.pi * f0 * (t - t0)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def _envelope(sig):
    """Analytic-signal envelope via the Hilbert transform (for a clean arrival-time pick)."""
    x = np.asarray(sig, dtype=float)
    N = len(x)
    F = np.fft.fft(x)
    hh = np.zeros(N)
    hh[0] = 1.0
    hh[1 : (N + 1) // 2] = 2.0
    if N % 2 == 0:
        hh[N // 2] = 1.0
    return np.abs(np.fft.ifft(F * hh))


def _run_traces(solver, ops, *, source, receivers, nsteps, f0):
    """Propagate a Ricker-sourced wavefield and return the recorded solid-velocity traces at receivers."""
    dt = solver.dt
    t0 = 1.5 / f0
    state = solver.zeros(ops)
    traces = {r: [] for r in receivers}
    for it in range(nsteps):
        t = it * dt
        amp = _ricker(t, f0, t0)
        src = {k: amp * v for k, v in source.items()}
        state = solver.step(state, ops, source=src)
        vfield = solver.solid_velocity(state).detach().numpy()
        qfield = solver.fluid_velocity(state).detach().numpy()
        for r in receivers:
            traces[r].append((vfield[r], qfield[r]))
    return {r: np.asarray(v) for r, v in traces.items()}


class TestGassmannModuli(unittest.TestCase):
    def test_moduli_closed_form(self):
        """The derived Biot-Gassmann moduli match the closed-form definitions."""
        mod = gassmann_moduli(k_solid=36e9, k_fluid=2.2e9, k_dry=9e9, mu=7e9, phi=0.25)
        alpha = 1.0 - 9e9 / 36e9
        M = 1.0 / (0.25 / 2.2e9 + (alpha - 0.25) / 36e9)
        k_sat = 9e9 + alpha**2 * M
        self.assertAlmostEqual(mod["alpha"], alpha, places=10)
        self.assertAlmostEqual(mod["M"], M, delta=1.0)
        self.assertAlmostEqual(mod["k_sat"], k_sat, delta=1.0)
        self.assertAlmostEqual(mod["H"], k_sat + 4.0 / 3.0 * 7e9, delta=1.0)

    def test_saturated_stiffer_than_dry(self):
        """Gassmann fluid substitution raises the bulk modulus (undrained > drained)."""
        mod = gassmann_moduli(k_solid=36e9, k_fluid=2.2e9, k_dry=9e9, mu=7e9, phi=0.25)
        self.assertGreater(mod["k_sat"], 9e9)


class TestFastPVelocity(unittest.TestCase):
    def test_fast_p_matches_biot_gassmann(self):
        """The measured fast-P phase velocity matches sqrt(H/rho) to ~2-3% (low-frequency undrained)."""
        ops = make_ops()
        vp_ref = biot_gassmann_velocity(
            k_solid=ROCK["k_solid"],
            k_fluid=ROCK["k_fluid"],
            k_dry=ROCK["k_dry"],
            mu=ROCK["mu"],
            phi=ROCK["phi"],
            rho_solid=ROCK["rho_solid"],
            rho_fluid=ROCK["rho_fluid"],
        )
        n, dx = 4000, 1.0
        solver = BiotPoroelastic1D(n, dt=0.35 * dx / 3300.0, spacing=dx, **ROCK)
        # the solver reports the same reference velocity
        self.assertAlmostEqual(solver.vp, vp_ref, delta=1.0)

        f0 = 20.0  # low frequency: fast-P should sit at the Biot-Gassmann limit
        xs, r1, r2 = 300, 800, 1600
        traces = _run_traces(
            solver,
            ops,
            source=solver.solid_source(xs, amplitude=1e-3),
            receivers=[r1, r2],
            nsteps=6000,
            f0=f0,
        )
        v1 = traces[r1][:, 0]
        v2 = traces[r2][:, 0]
        p1 = int(np.argmax(_envelope(v1)))
        p2 = int(np.argmax(_envelope(v2)))
        v_meas = (r2 - r1) * dx / ((p2 - p1) * solver.dt)
        rel_err = abs(v_meas - vp_ref) / vp_ref
        self.assertLess(rel_err, 0.03, f"fast-P {v_meas:.1f} vs Gassmann {vp_ref:.1f} (rel {rel_err:.4f})")


class TestSlowWave(unittest.TestCase):
    def test_slow_wave_exists_and_is_diffusive(self):
        """A pore-pressure drive excites the slow (Biot) P wave, which is diffusive at low frequency.

        At seismic frequencies (far below the Biot transition ~20 kHz for this rock) the slow wave does not
        radiate: the pore-pressure disturbance stays localized around the source with a sub-wavelength
        penetration depth (a diffusion length ``sqrt(D/omega)``, ``D = M k / eta``), while the solid velocity
        is carried away at the fast-P speed. We verify both: the pore pressure is huge at the source and
        essentially zero where the fast-P front has reached, whereas the solid velocity is present there.
        """
        ops = make_ops()
        n, dx = 3000, 1.0
        solver = BiotPoroelastic1D(n, dt=0.35 * dx / 3300.0, spacing=dx, **ROCK)
        f0 = 20.0  # far below the Biot transition frequency: the slow wave is diffusive
        xs = n // 2
        nsteps = 2500
        t0 = 1.5 / f0
        state = solver.zeros(ops)
        for it in range(nsteps):
            src = solver.pressure_source(xs, amplitude=_ricker(it * solver.dt, f0, t0))
            state = solver.step(state, ops, source=src)
        v, q, sigma, pf = (x.detach().numpy() for x in solver.unpack(state))

        # the fast-P front has travelled this far by the end of the run
        front = int(solver.vp * nsteps * solver.dt)
        self.assertGreater(front, 400)  # the front is well inside the grid

        pf_at_source = np.abs(pf[xs - 20 : xs + 20]).max()
        pf_at_front = np.abs(pf[xs + front - 20 : xs + front + 20]).max()
        v_at_front = np.abs(v[xs + front - 20 : xs + front + 20]).max()

        # (a) the pore-pressure disturbance is trapped near the source (diffusive: it does not radiate).
        self.assertGreater(pf_at_source, 1e4 * pf_at_front)

        # (b) the diffusive penetration depth is sub-wavelength: pf falls to 1/e within a few cells.
        amp = np.abs(pf[xs : xs + 200])
        decayed = np.where(amp < amp[0] / np.e)[0]
        self.assertLessEqual(int(decayed[0]), 5)

        # (c) the fast-P wave (solid velocity) IS present at the front, confirming two distinct modes.
        self.assertGreater(v_at_front, 0.0)
        self.assertTrue(np.isfinite(v_at_front))

    def test_two_characteristic_speeds(self):
        """The undrained characteristic system has two real P speeds: a fast one at ~vp and a faster c_max
        bracketing it, and the slow branch is a distinct, much lower root."""
        solver = BiotPoroelastic1D(64, dt=1e-4, spacing=1.0, **ROCK)
        R = np.array([[solver.H, solver.C], [solver.C, solver.M]])
        T = np.array([[solver.rho, solver.rho_f], [solver.rho_f, solver.m]])
        speeds = np.sort(np.sqrt(np.linalg.eigvals(np.linalg.solve(T, R)).real))
        slow, fast = speeds
        # two distinct real speeds; the slow one is far below the fast one
        self.assertLess(slow, 0.5 * fast)
        # fast branch brackets the Biot-Gassmann velocity from above (undrained coupling stiffens it)
        self.assertGreaterEqual(fast + 1e-6, solver.vp)
        self.assertAlmostEqual(fast, solver.c_max, delta=1.0)


class TestDifferentiable(unittest.TestCase):
    def test_gradient_flows_through_step(self):
        """A recorded trace is differentiable w.r.t. an initial-condition tensor through the ops step."""
        import torch

        ops = make_ops()
        n = 200
        solver = BiotPoroelastic1D(n, dt=1e-4, spacing=1.0, **ROCK)
        amp = torch.tensor(1.0e-3, dtype=torch.float64, requires_grad=True)
        # seed a solid-velocity pulse scaled by the differentiable amplitude, near the observation node so
        # the wave reaches it within the short run (a few cells at ~3200 m/s over 30 steps of dt=1e-4).
        seed = torch.zeros(n, dtype=torch.float64)
        seed[100] = 1.0
        v = amp * seed
        state = solver.pack(v, torch.zeros(n), torch.zeros(n), torch.zeros(n))
        for _ in range(30):
            state = solver.step(state, ops)
        obs = solver.solid_velocity(state)[104]
        obs.backward()
        self.assertIsNotNone(amp.grad)
        self.assertTrue(torch.isfinite(amp.grad))
        self.assertNotEqual(float(amp.grad), 0.0)


if __name__ == "__main__":
    unittest.main()
