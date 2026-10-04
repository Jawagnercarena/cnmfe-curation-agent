# Rules for Claude on this project

**Moved.** As of 2026-09-22 the standing rules live in [`../CLAUDE.md`](../CLAUDE.md)
at the repository root, where Claude Code loads them automatically in every session.

On another machine nothing needs copying: clone the repo and the root file is
picked up. Machine-specific paths are re-derived through `agent/local_config.py`
(see section 5 of the root file); set them in `agent/.env`, never in the rules.

Operator procedures with side effects are skills under `.claude/skills/` and are
invoked by name (for example `/retrain-area`).

The pre-2026-09-22 long-form version of this file, with incident history, is in
git history (`git log -- docs/CLAUDE_RULES.md`).
