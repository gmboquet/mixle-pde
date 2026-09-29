"""Commit authorship in the release range is this project's own, and it is checked, not assumed.

The same rule and the same script as mixle core (``scripts/check_commit_attribution.py``): no tool
co-author trailer, no vendor name, no generated-by banner, no bare "AI" in any commit message of the
release range. The range starts at the commit this line was cut from.
"""

from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_BASE = "19797870"


def _module():
    path = ROOT / "scripts" / "check_commit_attribution.py"
    spec = importlib.util.spec_from_file_location("_check_commit_attribution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _has_range() -> bool:
    try:
        subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--verify", RELEASE_BASE + "^{commit}"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return True


class RuleTest(unittest.TestCase):
    def setUp(self):
        self.module = _module()

    def _markers(self, text):
        return [why for pattern, why in self.module.MARKERS if pattern.search(text)]

    def test_tool_attribution_is_refused(self):
        for text in (
            "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>",
            "Generated with [Claude Code](https://claude.ai/code)",
            "Assisted-By: ChatGPT <someone@openai.com>",
            "written with GitHub Copilot",
            "drafted by Codex and reviewed by hand",
            "a Gemini pass over the docstrings",
            "answers checked against gpt-4o",
            "Ten AI adversarial reviews of the notebooks and examples",
        ):
            self.assertTrue(self._markers(text), text)

    def test_subject_matter_is_not_attribution(self):
        for text in (
            "regenerated with zero drift -- no new top-level names, only methods on classes",
            "OpenAICompatLLM keeps the provider-neutral surface; ai_operability stays a module name",
            "raise the gain floor; the retained receipts say why",
        ):
            self.assertEqual(self._markers(text), [], text)

    def test_owner_is_allowed_in_every_role(self):
        for role in ("author", "committer", "co-author"):
            self.assertIn(self.module.OWNER, self.module.ALLOWED_BY_ROLE[role])


class DetectionTest(unittest.TestCase):
    def test_a_violating_history_is_detected(self):
        """A gate that cannot fail proves nothing: build a two-commit history whose second commit carries a
        tool trailer and a vendor name, and check the range trips."""
        module = _module()
        with tempfile.TemporaryDirectory() as tmp:
            env = {"GIT_AUTHOR_NAME": "Grant Boquet", "GIT_AUTHOR_EMAIL": "grant.boquet@gmail.com",
                   "GIT_COMMITTER_NAME": "Grant Boquet", "GIT_COMMITTER_EMAIL": "grant.boquet@gmail.com",
                   "HOME": tmp, "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}
            run = lambda *a: subprocess.run(["git", "-C", tmp, *a], check=True, capture_output=True, env=env)
            run("init", "-q", "-b", "main")
            Path(tmp, "a").write_text("a")
            run("add", "a"); run("commit", "-q", "-m", "clean start")
            base = subprocess.run(["git", "-C", tmp, "rev-parse", "HEAD"], capture_output=True, text=True, env=env).stdout.strip()
            Path(tmp, "b").write_text("b")
            run("add", "b")
            run("commit", "-q", "-m", "add b\n\nCo-Authored-By: Claude <noreply@anthropic.com>")
            saved = module.subprocess.run
            module.subprocess.run = lambda cmd, **kw: saved(["git", "-C", tmp, *cmd[1:]], **kw)
            try:
                found = module.violations(base, "HEAD")
            finally:
                module.subprocess.run = saved
        self.assertTrue(any("trailer" in line for line in found), found)
        self.assertTrue(any("names Claude" in line for line in found), found)


@unittest.skipUnless(_has_range(), "needs a git checkout that contains the release base")
class ReleaseRangeTest(unittest.TestCase):
    def test_the_release_range_claims_only_this_project_s_authorship(self):
        found = _module().violations(RELEASE_BASE, "HEAD")
        self.assertEqual(found, [], "\n".join(found))


if __name__ == "__main__":
    unittest.main()
