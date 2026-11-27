---
name: ship-branch
description: Ship the committed work on the current branch to main in one go - pre-push checks, push, open (or reuse) the PR, merge it, verify the merge on GitHub, then move the central machine back to main by pointer move without touching a file.
disable-model-invocation: true
argument-hint: "[--no-merge]"
---

# Ship the current branch to main and return the machine to main

Manual-only. Invoking it is the user's go-ahead to push, open the PR and merge
it (merge commit, like PRs #1-#6). `--no-merge` stops after the PR is open.
It ships only what is already COMMITTED on the branch; it never commits,
stages or stashes anything, and uncommitted files (often another session's or a
watcher's) stay exactly as they are.

Why a pointer move: the watchers run from this working tree. A plain
`git checkout main` deletes the branch's files when main lacks them, and a
`git pull` on a stale main rewrote 403 files on 2026-11-22.
`agent/return_to_main.py` switches only when origin/main's tree equals HEAD's
tree, so no file on disk changes and the watchers need no pause.

Never: `git push --force`, `gh pr merge --admin`, squash/rebase unless the user
says so, deleting a branch (local or remote), `git checkout main` / `git pull`
on this machine, committing to main, bypassing hooks.

## 0. Preflight (read-only)

```bash
git status --porcelain=v1 -b
git fetch origin
git log --oneline origin/main..HEAD
git rev-list --left-right --count origin/main...HEAD
git diff --stat origin/main...HEAD
```

- On `main` or a detached HEAD: stop, nothing to ship.
- Behind origin/main (left count > 0): stop and tell the user. After a merge
  the trees would differ, and `return_to_main.py` would refuse anyway.
- Name every uncommitted path and say it is NOT shipped and stays on disk.
- Checks on what is being shipped (`origin/main...HEAD`), reported with output:
  - No secrets: no `.env` / credential file in `--name-only`; grep the added
    lines for key/token/password patterns, and never print a match's value.
  - No non-ASCII in added lines of `.py` / `.m` files (CLAUDE.md s3). Use
    `LC_ALL=C.UTF-8 grep -P '[^\x00-\x7F]'` on `git diff -U0` output, and
    confirm a "none" result by counting bytes with `od` (the plain `grep -P`
    has failed silently here on locale).
  - No large or binary data: every added blob is under 5 MB (`git diff --numstat`
    shows `-` for binaries).
  - `py_compile` every changed `.py`. Run each `agent/test_*.py` that the diff
    touches or whose module it changes. Run them with `PYTHON_EXE` and
    `</dev/null`, and show the RESULT lines.
  - If an exchange script changed (CLAUDE.md s1 list), grep it for
    delete/move verbs and confirm they act only on local paths.
- Any check fails: stop and report. Fixing it is a new commit, which needs the
  user's ask.

## 1. Push

```bash
git push -u origin <branch>
```

If this is rejected as non-fast-forward, stop: someone else pushed to the
branch. Never force.

## 2. Open the PR, or reuse one

```bash
gh pr list --head <branch> --state all --json number,state,url
```

If none is open, write the body to a scratchpad file and create the PR:

```bash
gh pr create --base main --head <branch> --title "<title>" --body-file <file>
```

The body has one section per commit (from its message), what was checked in
step 0, and what is deliberately not included. It ends with the PR
attribution line from the system reminder.

If `--no-merge` was given, stop here and report the URL.

## 3. Merge

```bash
gh pr merge <N> --merge
```

If the harness or GitHub refuses, give the user the PR URL and ask them to
merge it on GitHub. Then wait with a Monitor that polls
`gh api repos/{owner}/{repo}/pulls/<N> --jq .merged` every 60 s, for up to an
hour. Do not take the user's "merged" as proof; it was premature twice.

## 4. Verify the merge on GitHub

```bash
gh api repos/{owner}/{repo}/pulls/<N> --jq '.merged, .merge_commit_sha, .merged_at'
```

`.merged` must be `true`. Otherwise stop; the machine stays on the branch,
which is harmless.

## 5. Return the machine to main

```bash
"<PYTHON_EXE>" agent/return_to_main.py          # dry run, still fetches
"<PYTHON_EXE>" agent/return_to_main.py --apply
```

Read the dry run before applying: tree check equal, local main with no own
commits, the uncommitted files listed as "stays as it is".

| Exit | Meaning | What to do |
|---|---|---|
| 0 | `DONE` / `verified` | Done. |
| 2 | Refused, nothing changed | Report the listed paths. Main moved on, or the merge differs. A real checkout is the user's call; it changes files under the watchers. |
| 3 | Moved, but the after-snapshot differs | Check each listed path. A watcher writing at that moment is the usual cause. Do not move back on a hunch. |
| 1 | git error | Report it verbatim. |

## 6. Report and remember

Report:
- PR number and URL, and the merge commit;
- `HEAD` = `main` = `origin/main` (short SHA);
- the uncommitted files that stayed;
- that the branch was kept, locally and on GitHub.

Update the "Central machine branch state" section of memory
`project_rereview_and_quality.md` with the date and both SHAs. Also fix any
memory line that still says the machine is on the old branch.
