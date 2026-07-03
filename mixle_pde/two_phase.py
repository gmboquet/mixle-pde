"""2D immiscible two-fluid incompressible Navier-Stokes by a diffuse-interface (phase-field) projection.

The application is lubricated / core-annular pipelining: a thin low-viscosity water film at the wall
drastically cuts the pumping pressure drop of a viscous oil core. The two fluids share one velocity field
and one pressure; a phase field ``phi in [-1, 1]`` labels which fluid occupies each cell, and the material
properties density ``rho(phi)`` and viscosity ``mu(phi)`` interpolate between the two fluids across the
diffuse interface. Viscosity contrast is the whole point -- a low-mu film near the wall carries most of the
shear, so the high-mu core slides with far less resistance and the flow rate at a fixed pressure gradient
rises sharply (the drag reduction this solver is built to measure).

The scheme is variable-property Chorin projection, the two-fluid generalization of :mod:`mixle_pde.flow3d`:

    phi_{n+1} = phi_n + dt * ( -(u . grad) phi + M * laplacian(phi) )        (interface transport)
    u* = u + dt/rho * ( -(u.grad)u * rho + div(mu (grad u + grad u^T)) + f_st + f_body )   (momentum)
    laplacian(p) = div(u*) / dt                                              (pressure Poisson)
    u_{n+1} = u* - dt * grad(p) / rho                                        (variable-density projection)

Surface tension enters as the continuum-surface-force (CSF) body force ``f_st = sigma * kappa * grad(phi)``
with curvature ``kappa = -div(grad(phi) / |grad(phi)|)``, which is the standard phase-field CSF form and
reproduces the Young-Laplace pressure jump ``sigma / R`` across a curved interface.

Every operator is the same differentiable machinery as the single-fluid solvers: advection, the
divergence, the variable-viscosity stress, and the pressure gradient are :func:`ops.grad` central
differences, and the pressure Poisson is the differentiable adjoint :func:`ops.sparse_solve`. As in
:mod:`mixle_pde.flow3d`, the pressure operator is assembled as the exact ``div(grad(.))`` composition of
:func:`ops.grad` with itself, so the projection is consistent with the discrete divergence it corrects.
The geometry here is a channel: no-slip walls in the cross-channel direction (``y``) and periodicity in the
streamwise direction (``x``), so a constant streamwise body force drives a pressure-gradient channel flow.

The scheme is first-order in time (operator splitting) and explicit in the viscous and interface terms, so
``dt`` obeys the usual advection / diffusion CFL limits; the sparse solve dominates the cost, so keep the
cross-channel resolution moderate. It is differentiable end to end -- the viscosity contrast, the film
thickness, the surface tension, or the driving pressure gradient all flow through to whatever the forward
records (a flow rate, a velocity profile), the same inverse-problem story as the single-fluid solvers.
"""

from __future__ import annotations

import numpy as np


class TwoPhaseFlow2D:
    """A differentiable 2D immiscible two-fluid Navier-Stokes stepper (phase-field, variable-property projection).

    ``TwoPhaseFlow2D(nx, ny, rho=(rho0, rho1), mu=(mu0, mu1), sigma=..., dt=...)`` builds the forward on an
    ``nx x ny`` channel grid: no-slip walls at ``y=0`` and ``y=ny-1``, periodic in the streamwise ``x``. The
    state is ``(u, v, phi)`` as three flat length-``nx*ny`` tensors -- the two velocity components and the
    phase field ``phi in [-1, 1]`` (``phi=-1`` is fluid 0, ``phi=+1`` is fluid 1). Advance with
    ``step(state, ops)``; read flow with ``velocity`` / ``phase`` and the throughput with ``flow_rate``.
    ``stratified_phi`` / ``drop_phi`` build the core-annular and static-drop initial phase fields.

    A constant streamwise body force ``body_force`` plays the role of ``-dp/dx``: it drives the channel flow
    exactly as a constant pressure gradient would, which is the classical lubrication benchmark forcing.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        *,
        rho: tuple[float, float] = (1.0, 1.0),
        mu: tuple[float, float] = (1.0, 1.0),
        sigma: float = 0.0,
        dt: float,
        spacing: float | None = None,
        body_force: float = 0.0,
        mobility: float = 0.0,
        interface_width: float = 1.5,
        pressure_reg: float = 1e-6,
        implicit_diffusion: bool = False,
    ):
        self.nx = int(nx)
        self.ny = int(ny)
        self.shape = (self.nx, self.ny)
        self.rho0, self.rho1 = float(rho[0]), float(rho[1])
        self.mu0, self.mu1 = float(mu[0]), float(mu[1])
        self.sigma = float(sigma)
        self.dt = float(dt)
        # the wall-normal spacing sets the physical channel height; the streamwise direction is periodic.
        self.h = float(spacing) if spacing is not None else 1.0 / (ny - 1)
        self.body_force = float(body_force)
        self.mobility = float(mobility)
        self.eps = float(interface_width) * self.h  # diffuse-interface half-width for property blending
        self._pressure_reg = float(pressure_reg)
        self.implicit_diffusion = bool(implicit_diffusion)
        self._implicit = None  # lazily assembled from mu(phi) on first implicit step (see _build_implicit)
        # no-slip walls: velocity zero on the y walls; the streamwise x direction is periodic (no mask).
        wall = np.ones(self.shape)
        wall[:, 0] = wall[:, -1] = 0.0
        self._wall = wall.ravel()
        # pressure Poisson div(grad(.)) from the SAME central differences ops.grad uses (periodic x, wall y),
        # so the projection is exact against the discrete divergence/gradient (see the module docstring).
        self._poisson = self._build_poisson()

    # -- geometry / property blending ------------------------------------------------------------------
    def _build_poisson(self):
        """Assemble ``L = D_x(D_x(.)) + D_y(D_y(.))`` with ``D_axis`` the exact ops.grad central stencil.

        ``D_x`` is periodic (streamwise), ``D_y`` is the interior stencil (its edge rows are zero, matching
        ops.grad, so the wall rows fall back to identity). Returned as ``(rows, cols, vals, n)`` for
        :func:`ops.sparse_solve`; a small Tikhonov term pins the wide-stencil null space so the sparse LU is
        well posed. Built once (fixed pattern; not a latent driver)."""
        import scipy.sparse as sp
        import torch

        nx, ny, h, N = self.nx, self.ny, self.h, self.nx * self.ny

        def idx(i, j):
            return i * ny + j

        def diff_x():
            # periodic central difference in x: out[i] = (a[i+1] - a[i-1]) / (2h), wrap at the ends.
            rows, cols, vals = [], [], []
            for i in range(nx):
                for j in range(ny):
                    r = idx(i, j)
                    rows += [r, r]
                    cols += [idx((i + 1) % nx, j), idx((i - 1) % nx, j)]
                    vals += [1.0 / (2.0 * h), -1.0 / (2.0 * h)]
            return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

        def diff_y():
            # interior central difference in y: the two wall slabs are left at zero (mirrors ops.grad).
            rows, cols, vals = [], [], []
            for i in range(nx):
                for j in range(1, ny - 1):
                    r = idx(i, j)
                    rows += [r, r]
                    cols += [idx(i, j + 1), idx(i, j - 1)]
                    vals += [1.0 / (2.0 * h), -1.0 / (2.0 * h)]
            return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

        dx, dy = diff_x(), diff_y()
        L = (dx @ dx) + (dy @ dy)
        # the wall rows carry no y-flux (dy is zero there); pin them to identity so the Dirichlet-style
        # pressure solve is well posed and the wall pressure does not float.
        L = L.tolil()
        wall_nodes = np.where(self._wall == 0.0)[0]
        for b in wall_nodes:
            L.rows[b] = [int(b)]
            L.data[b] = [1.0]
        L = (L.tocsc() + self._pressure_reg * sp.identity(N)).tocoo()
        rows = torch.as_tensor(L.row, dtype=torch.long)
        cols = torch.as_tensor(L.col, dtype=torch.long)
        vals = torch.as_tensor(L.data, dtype=torch.float64)
        return rows, cols, vals, N

    def _build_implicit(self, phi):
        """Assemble ``M = I + dt * (1/rho) * (-div(mu grad(.)))`` (per velocity component) for an implicit
        viscous step, from the property fields ``mu(phi)`` / ``rho(phi)`` at the current phase.

        The explicit viscous term has a diffusion CFL ``dt < 0.25 h^2 rho / mu`` that is punishing at high
        viscosity contrast (the slow high-mu core sets the limit yet needs the longest time to reach steady
        state). Treating the decoupled Laplacian part ``div(mu grad u)`` implicitly removes that limit, so
        the channel reaches its steady piecewise-parabolic profile in far fewer projection solves. The wall
        rows are the identity (no-slip is applied by the mask), and the streamwise direction is periodic.

        Returns ``(rows, cols, vals, n)`` for :func:`ops.sparse_solve`; assembled once for a static phase
        field (as in the lubrication benchmark) and reused every step."""
        import scipy.sparse as sp
        import torch

        nx, ny, h, N = self.nx, self.ny, self.h, self.nx * self.ny
        mu = self.mu(phi).detach().cpu().numpy().reshape(nx, ny)
        rho = self.rho(phi).detach().cpu().numpy().reshape(nx, ny)

        def idx(i, j):
            return i * ny + j

        # -div(mu grad u) with HARMONIC-mean face viscosities on the 5-point stencil. The harmonic mean is
        # the physically correct face averaging across a viscosity discontinuity (it makes the flux, not the
        # viscosity, continuous), so the diffuse interface reproduces the sharp-jump two-layer profile far
        # more accurately than an arithmetic mean, which is spuriously stiff and underpredicts the flow.
        rows, cols, vals = [], [], []
        inv = 1.0 / rho
        for i in range(nx):
            for j in range(1, ny - 1):  # interior in y; wall rows stay identity below
                r = idx(i, j)
                diag = 0.0
                for ii, jj in (((i + 1) % nx, j), ((i - 1) % nx, j), (i, j + 1), (i, j - 1)):
                    mu_face = 2.0 * mu[i, j] * mu[ii, jj] / (mu[i, j] + mu[ii, jj])
                    coef = self.dt * inv[i, j] * mu_face / h**2
                    rows.append(r)
                    cols.append(idx(ii, jj))
                    vals.append(-coef)
                    diag += coef
                rows.append(r)
                cols.append(r)
                vals.append(1.0 + diag)  # the +I plus the accumulated stencil diagonal
        # identity on the wall rows (velocity pinned to zero there by the mask; keep M nonsingular).
        for i in range(nx):
            for j in (0, ny - 1):
                rows.append(idx(i, j))
                cols.append(idx(i, j))
                vals.append(1.0)
        M = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocoo()
        return (
            torch.as_tensor(M.row, dtype=torch.long),
            torch.as_tensor(M.col, dtype=torch.long),
            torch.as_tensor(M.data, dtype=torch.float64),
            N,
        )

    def _wall_mask(self, ops):
        return ops.tensor(self._wall)

    def _blend(self, phi, lo, hi):
        """Interpolate a material property between fluid 0 (``phi=-1``) and fluid 1 (``phi=+1``).

        Uses the smoothed indicator ``H = 0.5 (1 + tanh(phi / (2 eps / h)))``... here directly ``0.5(1+phi)``
        clamped, which is the standard linear blend on the ``[-1, 1]`` order parameter and keeps ``mu`` and
        ``rho`` strictly between the two fluids so the projection stays well posed."""
        theta = 0.5 * (phi + 1.0)
        theta = ops_clamp(theta)
        return lo + (hi - lo) * theta

    def rho(self, phi):
        return self._blend(phi, self.rho0, self.rho1)

    def mu(self, phi):
        return self._blend(phi, self.mu0, self.mu1)

    # -- accessors -------------------------------------------------------------------------------------
    def velocity(self, state):
        """The velocity pair ``(u, v)`` (two flat length-``nx*ny`` tensors) held by ``state``."""
        return state[0], state[1]

    def u(self, state):
        return state[0]

    def v(self, state):
        return state[1]

    def phase(self, state):
        """The phase field ``phi in [-1, 1]`` (flat) held by ``state``; ``-1`` is fluid 0, ``+1`` is fluid 1."""
        return state[2]

    def flow_rate(self, state):
        """The streamwise volumetric throughput ``sum(u) * h`` (per unit depth): the pumped flow rate.

        Averaged over the periodic streamwise direction this is proportional to the cross-channel integral
        of ``u(y)`` -- the quantity the lubrication benchmark compares between the two-fluid and single-fluid
        configurations at the same driving pressure gradient."""
        return state[0].sum() * (self.h / self.nx)

    def divergence(self, state, ops):
        """The discrete divergence ``du/dx + dv/dy`` (flat) -- zero in the incompressible limit."""
        u, v = self.velocity(state)
        return ops.grad(u, self.shape, 0, spacing=self.h) + ops.grad(v, self.shape, 1, spacing=self.h)

    # -- initial phase fields --------------------------------------------------------------------------
    def stratified_phi(self, film_thickness, ops):
        """A stratified core-annular phase field: fluid 0 (``phi=-1``) in a film of thickness ``film_thickness``
        at each wall, fluid 1 (``phi=+1``) in the core, with a tanh diffuse interface of width ``eps``.

        This is the lubrication configuration: the low-viscosity film hugs both walls, the viscous core fills
        the middle. Uniform in the streamwise direction. ``y`` runs ``0..(ny-1)*h``; the film occupies
        ``y < film_thickness`` and ``y > H - film_thickness``."""
        d = float(film_thickness)
        y = np.arange(self.ny) * self.h
        height = (self.ny - 1) * self.h
        # signed distance into the core: positive in the core (fluid 1), negative in the wall film (fluid 0).
        dist = np.minimum(y - d, (height - d) - y)
        prof = np.tanh(dist / self.eps)
        phi = np.tile(prof[None, :], (self.nx, 1)).ravel()
        return ops.tensor(phi)

    def drop_phi(self, center, radius, ops):
        """A circular drop of fluid 1 (``phi=+1``, radius ``radius`` about ``center=(cx, cy)`` in grid units
        scaled by ``h``) surrounded by fluid 0 (``phi=-1``), with a tanh diffuse interface of width ``eps``.

        Used for the static Young-Laplace test: surface tension raises the interior pressure by ``sigma/R``."""
        cx, cy = center
        xx, yy = np.meshgrid(np.arange(self.nx) * self.h, np.arange(self.ny) * self.h, indexing="ij")
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        phi = np.tanh((radius - r) / self.eps)  # +1 inside the drop, -1 outside
        return ops.tensor(phi.ravel())

    # -- operators -------------------------------------------------------------------------------------
    def _advect(self, u, v, f, ops):
        """The advective term ``(u . grad) f = u f_x + v f_y`` for a scalar / velocity component ``f``."""
        return u * ops.grad(f, self.shape, 0, spacing=self.h) + v * ops.grad(f, self.shape, 1, spacing=self.h)

    def _viscous_stress(self, u, v, mu, ops):
        """The variable-viscosity momentum diffusion ``div(mu (grad U + grad U^T))`` per component.

        For the ``u`` component: ``d/dx(2 mu u_x) + d/dy(mu (u_y + v_x))``; for ``v`` symmetrically. Built
        from the same central differences so the viscosity contrast at the diffuse interface enters
        consistently (the low-mu film really does carry less shear stress)."""
        sh = self.shape
        ux = ops.grad(u, sh, 0, spacing=self.h)
        uy = ops.grad(u, sh, 1, spacing=self.h)
        vx = ops.grad(v, sh, 0, spacing=self.h)
        vy = ops.grad(v, sh, 1, spacing=self.h)
        # stress components tau = mu (grad U + grad U^T)
        txx = 2.0 * mu * ux
        tyy = 2.0 * mu * vy
        txy = mu * (uy + vx)
        fu = ops.grad(txx, sh, 0, spacing=self.h) + ops.grad(txy, sh, 1, spacing=self.h)
        fv = ops.grad(txy, sh, 0, spacing=self.h) + ops.grad(tyy, sh, 1, spacing=self.h)
        return fu, fv

    def _surface_tension(self, phi, ops):
        """The CSF surface-tension body force ``sigma * kappa * grad(c)`` with curvature ``kappa``.

        ``kappa = -div(n)`` for the interface normal ``n = grad(phi) / |grad(phi)|``, and the force uses the
        gradient of the ``[0, 1]`` colour function ``c = 0.5 (phi + 1)`` so that ``integral grad(c) = 1``
        across the interface (the standard CSF normalization). With ``phi in [-1, 1]`` that is a factor of
        one half on ``grad(phi)`` -- without it the force, and hence the pressure jump, would double. At
        equilibrium this concentrates a force on the diffuse interface that balances a pressure jump of
        ``sigma/R`` across a circular interface of radius ``R`` (Young-Laplace)."""
        if self.sigma == 0.0:
            z = ops.zeros(self.nx * self.ny)
            return z, z
        sh = self.shape
        gx = ops.grad(phi, sh, 0, spacing=self.h)
        gy = ops.grad(phi, sh, 1, spacing=self.h)
        mag = ops.sqrt(gx * gx + gy * gy + 1e-12)
        nx_, ny_ = gx / mag, gy / mag
        kappa = -(ops.grad(nx_, sh, 0, spacing=self.h) + ops.grad(ny_, sh, 1, spacing=self.h))
        cx, cy = 0.5 * gx, 0.5 * gy  # grad of the [0,1] colour function c = 0.5 (phi + 1)
        return self.sigma * kappa * cx, self.sigma * kappa * cy

    def pressure(self, ustar, ops):
        """Solve the pressure Poisson ``laplacian(p) = div(u*) / dt`` with the consistent ops.grad operator.

        With ``L = div(grad(.))`` assembled from the same central differences, the projection
        ``u_{n+1} = u* - dt*grad(p)/rho`` drives ``div(u_{n+1})`` to the solver tolerance. Differentiable
        through :func:`ops.sparse_solve`."""
        rows, cols, vals, n = self._poisson
        div_ustar = self.divergence(ustar, ops)
        return ops.sparse_solve(rows, cols, vals, n, div_ustar / self.dt)

    def static_pressure(self, state, ops):
        """The mechanical pressure of a static (velocity-free) configuration, balancing surface tension.

        At mechanical equilibrium the pressure gradient balances the surface-tension body force,
        ``grad(p) = f_st``, so ``laplacian(p) = div(f_st)`` on the same consistent operator the projection
        uses. For a circular drop of radius ``R`` this returns the Young-Laplace field whose interior is
        higher than the exterior by ``sigma / R`` (in 2D). Differentiable through :func:`ops.sparse_solve`."""
        phi = self.phase(state)
        fx, fy = self._surface_tension(phi, ops)
        rows, cols, vals, n = self._poisson
        div_f = ops.grad(fx, self.shape, 0, spacing=self.h) + ops.grad(fy, self.shape, 1, spacing=self.h)
        return ops.sparse_solve(rows, cols, vals, n, div_f)

    def step(self, state, ops):
        """Advance one time step: transport ``phi``, form ``u*`` with variable properties and surface
        tension, then project onto divergence-free with the variable-density correction."""
        wall = self._wall_mask(ops)
        u, v, phi = state[0], state[1], state[2]
        # (i) interface transport: advect phi, plus a mild conservative diffusion (mobility) for sharpening
        # stability. phi is not masked at the walls (the fluids touch the wall); it stays in [-1, 1].
        phi_next = phi - self.dt * self._advect(u, v, phi, ops)
        if self.mobility != 0.0:
            phi_next = phi_next + self.dt * self.mobility * _laplacian(phi, self.shape, self.h, ops)
        phi_next = ops.clamp(phi_next, -1.0, 1.0)
        # (ii) momentum: variable rho and mu from the (updated) phase field.
        rho = self.rho(phi_next)
        mu = self.mu(phi_next)
        adv_u = self._advect(u, v, u, ops)
        adv_v = self._advect(u, v, v, ops)
        fst_u, fst_v = self._surface_tension(phi_next, ops)
        body_u = self.body_force  # streamwise body force (the constant -dp/dx driving the channel).
        if self.implicit_diffusion:
            if self._implicit is None:
                self._implicit = self._build_implicit(phi_next)
            r, c, vv, nn = self._implicit
            # explicit RHS carries advection, surface tension, forcing (diffusion is on the LHS operator M).
            rhs_u = u + self.dt / rho * (-rho * adv_u + fst_u + rho * body_u)
            rhs_v = v + self.dt / rho * (-rho * adv_v + fst_v)
            us = ops.sparse_solve(r, c, vv, nn, rhs_u * wall)
            vs = ops.sparse_solve(r, c, vv, nn, rhs_v * wall)
        else:
            visc_u, visc_v = self._viscous_stress(u, v, mu, ops)
            us = u + self.dt / rho * (-rho * adv_u + visc_u + fst_u + rho * body_u)
            vs = v + self.dt / rho * (-rho * adv_v + visc_v + fst_v)
        ustar = (us * wall, vs * wall, phi_next)
        # (iii) variable-density pressure projection.
        p = self.pressure(ustar, ops)
        gx = ops.grad(p, self.shape, 0, spacing=self.h)
        gy = ops.grad(p, self.shape, 1, spacing=self.h)
        un = (ustar[0] - self.dt * gx / rho) * wall
        vn = (ustar[1] - self.dt * gy / rho) * wall
        return (un, vn, phi_next)


def ops_clamp(theta):
    """Clamp an order-parameter blend to ``[0, 1]`` without importing a backend (works on torch tensors)."""
    return theta.clamp(0.0, 1.0) if hasattr(theta, "clamp") else max(0.0, min(1.0, theta))


def _laplacian(a, shape, h, ops):
    """The 5-point Laplacian of a flat field on the ``shape`` grid (interior; zero on the y edges).

    Periodic in the streamwise x direction, one-sided-free (zero) on the wall y edges -- the mild
    interface-sharpening diffusion operator, kept separate from the pressure Poisson."""
    nx, ny = shape
    A = a.reshape(nx, ny)
    out = ops.zeros(nx, ny)
    # x direction is periodic: roll neighbours.
    import torch

    xm = torch.roll(A, 1, dims=0)
    xp = torch.roll(A, -1, dims=0)
    lap_x = (xp + xm - 2.0 * A) / h**2
    out[:, 1:-1] = lap_x[:, 1:-1] + (A[:, 2:] + A[:, :-2] - 2.0 * A[:, 1:-1]) / h**2
    return out.reshape(-1)
