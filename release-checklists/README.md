# Release checklists

This folder is the tracked, public record of what has to be true before a `mixle-pde` cut ships. It
mirrors the process used by `mixle` core, whose family manifest (`manifests/family_release.json` in the
core repository) is the authority for the branch, version and pre-release identity this repository
carries: `mixle-pde` tracks core in lockstep, cut for cut.

- **`<version>.md`** (`0.8.3.md`, ...) -- one checklist per release line, created when the line is cut
  and updated in place *with evidence* as gates are verified. It stays in git history after the cut
  ships, so anyone can see exactly what was checked, how, and when.

This is a checklist of **gates**, not a task list. Statuses use core's vocabulary: `DONE` (verified,
evidence in the row), `IMPLEMENTED` (the gate exists and has not yet been measured on the tip that will
be tagged), `HOSTED` (waits on a hosted run), `EXTERNAL` (waits on a person), `EXCLUDED` (does not apply,
with the reason). Nothing is tagged while a pre-publication gate is open, and nothing is tagged before
core's own tag for the same version.

Two rules apply to every commit on the release line, enforced in CI by `scripts/check_commit_attribution.py`
(the same file, with the same rules, as core): the author and committer are this project's own, and no
commit message carries a tool co-author trailer, a vendor name (Anthropic, Claude, ChatGPT, Codex, Gemini,
Copilot, a GPT model), a generated-by banner, or the bare word "AI".
