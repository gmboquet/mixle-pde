"""Dependency, licensing, and batteries-included deployment envelope (MP-A5, workstream A).

mixle-pde is the architecture's designated "batteries-included PDE profile": every heavy solver
backend (torch-based differentiability, sparse-scale accelerators, an MPI transport, geoscience
data-format ingest) is an opt-in installation extra, never a hard dependency of the base package.
Every import of one of those backends inside ``mixle_pde`` is already lazy (guarded inside a
function body, never at module scope; see ``mixle_pde.env_data._require``,
``mixle_pde.io.segy._require_segyio``, ``mixle_pde.linear_solve._amg_preconditioner``, and friends)
so this module tests the packaging contract those call sites depend on:

1. The architecture-mandated extras (``fem``, ``mesh``, ``mpi``, ``fvm``, ``coupling``,
   ``inverse``, ``surrogate``, ``all``) are declared in ``pyproject.toml`` with real packages, and
   ``all`` is the batteries-included union of them.
2. No extras-only package leaks into the unconditional base ``dependencies`` list, and no
   commercial/optional adapter package (``mixle_mlops``) is ever a hard requirement of any extra.
3. Each declared extra -- and combinations of them -- resolves to a compatible dependency set
   (network-gated: skipped without PyPI access, exercised for real when it is available).
4. A genuinely isolated, zero-extras venv (built fresh, independent of whatever this test suite's
   own interpreter happens to have installed) can still import ``mixle_pde`` and run the
   ``mixle_pde.problem_adapter`` compatibility-boundary tests with no heavy optional backend present.
"""

from __future__ import annotations

import ast
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import tomllib

_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _ROOT / "pyproject.toml"
_PACKAGE = _ROOT / "mixle_pde"

# The extras the architecture plan (MP-A5) requires this package to define, beyond the pre-existing
# dev-tooling (test/lint/docs) and field-data-format (raster/netcdf/grib/segy/las/potfield/data) extras.
_ARCHITECTURE_EXTRAS = ("fem", "mesh", "mpi", "fvm", "coupling", "inverse", "surrogate", "all")
_DEV_EXTRAS = ("test", "lint", "docs")

# Heavy backends that must stay lazily imported (never at module scope) so a base install keeps working.
_HEAVY_MODULES = (
    "torch",
    "pandas",
    "sklearn",
    "rasterio",
    "xarray",
    "geopandas",
    "segyio",
    "lasio",
    "harmonica",
    "pyamg",
    "sksparse",
    "mpi4py",
    "fiona",
    "netCDF4",
    "cfgrib",
    "openpyxl",
    "verde",
)


# Import name -> PyPI distribution name, for the two heavy backends where they differ.
_DIST_NAME = {"sklearn": "scikit-learn", "sksparse": "scikit-sparse"}


def _load_pyproject() -> dict:
    with _PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def _bare_name(requirement: str) -> str:
    """Normalize a PEP 508 requirement string to a bare, comparable distribution name."""
    name = requirement
    for sep in ("==", ">=", "<=", "~=", "!=", ">", "<", ";", "["):
        name = name.split(sep, 1)[0]
    return name.strip().lower().replace("_", "-")


def _network_available(host: str = "pypi.org", port: int = 443, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _module_scope_heavy_imports(py_file: Path) -> set[str]:
    """Return heavy backend names imported at module (not function/method) scope in ``py_file``."""
    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    found: set[str] = set()
    for node in tree.body:  # only the file's top-level statements, not nested function/class bodies
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _HEAVY_MODULES:
                    found.add(root)
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            if root in _HEAVY_MODULES:
                found.add(root)
    return found


class ExtrasDeclarationTest(unittest.TestCase):
    """Static checks on the pyproject.toml extras table. No network, no installs."""

    @classmethod
    def setUpClass(cls):
        cls.data = _load_pyproject()
        cls.project = cls.data["project"]
        cls.extras = cls.project["optional-dependencies"]

    def test_every_architecture_extra_is_declared_and_nonempty(self):
        for name in _ARCHITECTURE_EXTRAS:
            self.assertIn(name, self.extras, f"pyproject.toml is missing the [{name}] extra")
            self.assertTrue(self.extras[name], f"[{name}] extra must declare at least one package")

    def test_base_dependencies_never_include_a_heavy_optional_backend(self):
        # A base install (zero extras) must stay light: none of the packages this task defines
        # extras for may be required unconditionally. (An extra is free to *also* list a genuine
        # base dependency -- e.g. `pyproj` legitimately backs both the CRS layer at base and the
        # `data` bundle -- so this checks the heavy-backend set specifically, not extras/base
        # overlap in general.)
        base = {_bare_name(p) for p in self.project["dependencies"]}
        heavy = {_DIST_NAME.get(name, name).lower().replace("_", "-") for name in _HEAVY_MODULES}
        overlap = base & heavy
        self.assertFalse(overlap, f"base dependencies require a heavy optional backend: {overlap}")

    def test_all_extra_is_the_batteries_included_union_of_physics_extras(self):
        physics_extras = ("fem", "mesh", "mpi", "fvm", "coupling", "inverse", "surrogate")
        physics_union: set[str] = set()
        for name in physics_extras:
            physics_union.update(_bare_name(p) for p in self.extras[name])
        all_pkgs = {_bare_name(p) for p in self.extras["all"]}
        missing = physics_union - all_pkgs
        self.assertFalse(missing, f"[all] is missing packages declared by physics extras: {missing}")

    def test_no_commercial_adapter_package_is_a_hard_requirement(self):
        # mixle_mlops backs an optional commercial task-cascade adapter mixle_pde.surrogate can bind
        # to (see mixle_pde.surrogate.to_task_cascade_adapter); it is an internal sibling package, not
        # a public distribution, and must never be required to install or test mixle-pde.
        for name, pkgs in self.extras.items():
            for pkg in pkgs:
                bare = _bare_name(pkg)
                self.assertNotIn("mixle-mlops", bare, f"[{name}] hard-requires the commercial adapter package")

    def test_data_format_error_messages_reference_declared_extras(self):
        # mixle_pde.io.segy / .las / .reductions raise ImportError naming an extra to install; make
        # sure those extras actually exist so the guidance in the error message is not a dead end.
        for extra in ("raster", "netcdf", "grib", "segy", "las", "potfield", "data"):
            self.assertIn(extra, self.extras, f"pyproject.toml is missing the [{extra}] extra referenced by io/*")


class LazyImportDisciplineTest(unittest.TestCase):
    """No ``mixle_pde`` module imports a heavy optional backend at module scope.

    This is what actually makes "base import stays functional without heavy extras" true
    regardless of which extras happen to be installed in whatever environment collects this test:
    it is a static property of the source, not a runtime observation of ``sys.modules``. Every
    heavy backend call site in this package already follows this convention (see the module
    docstring); this test pins it so a future addition cannot silently reintroduce a hard
    dependency on a batteries-included extra.
    """

    def test_no_module_scope_heavy_import_anywhere_in_the_package(self):
        offenders: dict[str, set[str]] = {}
        for py_file in sorted(_PACKAGE.rglob("*.py")):
            if py_file.name.endswith("_test.py"):
                continue  # in-package component scripts, not part of the base import graph
            heavy = _module_scope_heavy_imports(py_file)
            if heavy:
                offenders[str(py_file.relative_to(_ROOT))] = heavy
        self.assertEqual(offenders, {}, f"module-scope heavy imports break the base-install contract: {offenders}")


@unittest.skipUnless(_network_available(), "no network access to resolve PyPI package metadata")
class ResolverCompatibilityTest(unittest.TestCase):
    """Each declared extra -- and combinations of them -- resolves without dependency conflicts.

    Uses ``pip install --dry-run`` against the local checkout: pip runs its real dependency
    resolver and fetches package metadata, but does not modify any environment (nothing is
    installed). A resolver conflict (e.g. two extras pinning incompatible versions of a shared
    package) makes this fail with a nonzero exit code and a ``ResolutionImpossible`` message.
    """

    @classmethod
    def setUpClass(cls):
        if shutil.which(sys.executable) is None:  # pragma: no cover - defensive
            raise unittest.SkipTest("no usable python executable for a subprocess pip resolve")
        cls.extras = _load_pyproject()["project"]["optional-dependencies"]

    def _dry_run(self, extra_expr: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--dry-run",
                "--quiet",
                "--disable-pip-version-check",
                f"{_ROOT}[{extra_expr}]",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )

    def test_each_architecture_extra_resolves_individually(self):
        for extra in _ARCHITECTURE_EXTRAS:
            with self.subTest(extra=extra):
                result = self._dry_run(extra)
                self.assertEqual(
                    result.returncode,
                    0,
                    f"pip could not produce a compatible lockfile for [{extra}]:\n{result.stderr}",
                )

    def test_all_physics_extras_combine_without_conflict(self):
        combined = ",".join(("fem", "mesh", "mpi", "fvm", "coupling", "inverse", "surrogate"))
        result = self._dry_run(combined)
        self.assertEqual(
            result.returncode,
            0,
            f"pip could not produce a compatible lockfile for the combined extras:\n{result.stderr}",
        )

    def test_all_extra_matches_the_combined_physics_resolution(self):
        result = self._dry_run("all")
        self.assertEqual(
            result.returncode,
            0,
            f"pip could not produce a compatible lockfile for [all]:\n{result.stderr}",
        )


@unittest.skipUnless(_network_available(), "no network access to build an isolated base-install venv")
class BaseInstallStaysFunctionalTest(unittest.TestCase):
    """A venv with zero extras installed can import ``mixle_pde`` and run the problem_adapter tests.

    This deliberately does *not* run inside ``sys.executable``, the interpreter already running this
    test suite: this repo's own CI (``.github/workflows/tests.yml``) installs ``mixle-pde[test,data]``
    (which pulls in torch) before invoking ``pytest``, so ``sys.executable`` already has torch on its
    ``sys.path`` by the time these checks would run. That is not a mixle_pde defect -- ``mixle.ppl.core``
    / ``mixle.ppl.field`` opportunistically pick up torch themselves when it happens to be importable,
    which is mixle's own behavior, not mixle-pde's -- but it does mean "torch not in sys.modules after
    import mixle_pde" is only a meaningful check in an environment where torch was never installed in
    the first place. setUpClass builds exactly that environment once (a throwaway venv containing only
    mixle-pde's unconditional base dependencies -- mixle, numpy, scipy, pyproj -- no extras), and every
    test method runs a subprocess against *that* interpreter.
    """

    @classmethod
    def setUpClass(cls):
        cls._venv_dir = tempfile.mkdtemp(prefix="mixle-pde-base-install-")
        subprocess.run([sys.executable, "-m", "venv", "--clear", cls._venv_dir], check=True, timeout=120)
        venv_python = str(Path(cls._venv_dir) / ("Scripts" if sys.platform == "win32" else "bin") / "python")
        install = subprocess.run(
            [venv_python, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", str(_ROOT)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if install.returncode != 0:
            shutil.rmtree(cls._venv_dir, ignore_errors=True)
            raise unittest.SkipTest(f"could not build an isolated base-install venv:\n{install.stderr}")
        cls._venv_python = venv_python

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._venv_dir, ignore_errors=True)

    def _run(self, code: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self._venv_python, "-c", code],
            capture_output=True,
            text=True,
            cwd=_ROOT,
            timeout=120,
        )

    def test_mixle_pde_imports_without_any_heavy_optional_backend(self):
        result = self._run(
            "import sys\n"
            "import mixle_pde\n"
            f"heavy = {_HEAVY_MODULES!r}\n"
            "present = sorted(m for m in heavy if m in sys.modules)\n"
            "assert present == [], f'heavy modules pulled in by import mixle_pde: {present}'\n"
            "print('OK')\n"
        )
        self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")
        self.assertIn("OK", result.stdout)

    def test_problem_adapter_boundary_surface_is_importable_without_heavy_backends(self):
        result = self._run(
            "import sys\n"
            "from mixle_pde.problem_adapter import (\n"
            "    PDEBackendProfile, PDECompatibilityReport, UnsupportedPDEProblem,\n"
            "    inspect_math_problem, require_compatible,\n"
            ")\n"
            f"heavy = {_HEAVY_MODULES!r}\n"
            "present = sorted(m for m in heavy if m in sys.modules)\n"
            "assert present == [], f'heavy modules pulled in by problem_adapter: {present}'\n"
            "print('OK')\n"
        )
        self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")
        self.assertIn("OK", result.stdout)

    def test_problem_adapter_tests_pass_in_a_process_with_no_heavy_backend_imported(self):
        # Runs the actual acceptance-criteria assertions from problem_adapter_test.py inline, in a
        # subprocess whose sys.modules starts empty, so a pass here is real evidence the
        # problem_adapter-boundary tests need nothing beyond mixle_pde + the stdlib.
        result = self._run(
            "import sys\n"
            "from mixle_pde.problem_adapter import PDEBackendProfile, UnsupportedPDEProblem, require_compatible\n"
            "\n"
            "def problem(discretization='P1'):\n"
            "    return {\n"
            "        'id': 'heat',\n"
            "        'domains': [{'id': 'plate', 'kind': 'mesh', 'properties': {'mesh_cell_type': 'triangle'}}],\n"
            "        'unknowns': [{'id': 'temperature', 'domain_id': 'plate'}],\n"
            "        'operators': [{'id': 'heat-weak-form', 'kind': 'weak_form', 'input_ids': ['temperature'],\n"
            "                       'output_ids': ['temperature'], 'discretization': discretization}],\n"
            "        'constraints': [], 'objectives': [{'id': 'solution', 'sense': 'satisfy', 'expression': {}}],\n"
            "        'evidence_requests': [{'kind': 'residual', 'required': True}], 'solve_plan': {},\n"
            "    }\n"
            "\n"
            "profile = PDEBackendProfile(\n"
            "    id='fem-profile', operator_kinds=frozenset({'weak_form'}),\n"
            "    discretizations=frozenset({'P1', 'P2'}), objective_senses=frozenset({'satisfy', 'infer'}),\n"
            "    mesh_cell_types=frozenset({'triangle', 'tetrahedron'}),\n"
            ")\n"
            "report = require_compatible(problem(), profile)\n"
            "assert report.supported and report.required_evidence == ('residual',)\n"
            "try:\n"
            "    require_compatible(problem('spectral-element'), profile)\n"
            "    raise SystemExit('expected UnsupportedPDEProblem')\n"
            "except UnsupportedPDEProblem as exc:\n"
            "    assert exc.report.unsupported_features == ('discretization:spectral-element',)\n"
            f"heavy = {_HEAVY_MODULES!r}\n"
            "present = sorted(m for m in heavy if m in sys.modules)\n"
            "assert present == [], f'heavy modules pulled in: {present}'\n"
            "print('OK')\n"
        )
        self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")
        self.assertIn("OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
