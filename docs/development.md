# Development

Work through PRs against `release/0.8.0`. Preserve existing APIs, add negative tests for every adapter validator, and
run focused legacy parity plus canonical conformance tests. Do not use package-internal imports from sibling projects.

Run Ruff, focused Pytest with `-n 0`, package build, and clean-wheel runtime locally. Full solver matrices remain CI or
release gates unless the change requires them. Record immutable merge and test evidence in Mixle Status.
