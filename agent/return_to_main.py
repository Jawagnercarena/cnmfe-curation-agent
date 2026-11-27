"""Move this working tree from a merged branch back to main without touching a file.

The central machine runs the watchers from this working tree, so a branch switch
must not rewrite, add or delete a single file.  A plain `git checkout main` after
a merge can do exactly that when local main is behind (2026-11-22: 403 files) or
when main lacks the branch's commits (it deletes them).  This script only moves
pointers, and only when that is provably file-neutral:

  1. on a named branch other than main (detached HEAD refused)
  2. `git fetch <remote>`  (writes inside .git only)
  3. local main has no commits of its own (not in <remote>/main)
  4. <remote>/main's TREE equals HEAD's tree: every tracked file is identical,
     so the switch changes nothing on disk.  Otherwise it refuses and lists the
     paths that differ (PR not merged yet, or main moved on since the merge).
  5. snapshot `git status --porcelain --ignored` plus sha256 of every modified
     and untracked file; then `git branch -f main <sha>` and `git checkout main`;
     then the same snapshot again, compared entry by entry.

Dry run unless --apply (the dry run still fetches).  The branch itself is kept,
locally and on the remote; nothing is deleted.  Merge verification on GitHub is
the caller's job (the /ship-branch skill); here the tree check is the gate.

Exit codes: 0 moved (or would move, or already on main); 2 refused, nothing
changed; 3 moved but the after-snapshot differs (investigate); 1 git error.
"""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

HASH_CAP = 256 * 1024 * 1024  # larger untracked files: size + mtime only


class GitError(Exception):
    pass


def git(repo, *args, check=True):
    p = subprocess.run(["git", "-C", str(repo)] + list(args),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and p.returncode != 0:
        raise GitError("git " + " ".join(args) + " failed ("
                       + str(p.returncode) + "): "
                       + p.stderr.decode("utf-8", "replace").strip())
    return p


def out(repo, *args, check=True):
    return git(repo, *args, check=check).stdout.decode("utf-8", "replace").strip()


def status_entries(repo, *extra):
    """Parse `git status --porcelain=v1 -z` into sorted (XY, path) tuples."""
    raw = git(repo, "status", "--porcelain=v1", "-z", *extra).stdout
    parts = raw.split(b"\0")
    entries = []
    i = 0
    while i < len(parts):
        rec = parts[i]
        i += 1
        if not rec:
            continue
        xy = rec[:2].decode("ascii", "replace")
        path = rec[3:].decode("utf-8", "surrogateescape")
        if xy[0] in "RC":  # rename/copy: the next record is the source path
            src = parts[i].decode("utf-8", "surrogateescape") if i < len(parts) else ""
            i += 1
            path = path + " <- " + src
        entries.append((xy, path))
    return sorted(entries)


def file_fingerprint(repo, rel):
    p = Path(repo) / rel
    if p.is_dir():
        return ("dir",)
    if not p.exists():
        return ("absent",)
    st = p.stat()
    if st.st_size > HASH_CAP:
        return ("size+mtime", st.st_size, st.st_mtime_ns)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return ("sha256", st.st_size, h.hexdigest())


def snapshot(repo):
    ignored = status_entries(repo, "--ignored")
    changed = status_entries(repo, "-uall")
    prints = {}
    for xy, path in changed:
        rel = path.split(" <- ")[0]
        prints[rel] = file_fingerprint(repo, rel)
    return {"status_ignored": ignored, "status_changed": changed, "prints": prints}


def compare(before, after):
    diffs = []
    for key in ("status_ignored", "status_changed"):
        b, a = set(before[key]), set(after[key])
        for e in sorted(b - a):
            diffs.append(key + " gone: " + e[0] + " " + e[1])
        for e in sorted(a - b):
            diffs.append(key + " new:  " + e[0] + " " + e[1])
    for rel in sorted(set(before["prints"]) | set(after["prints"])):
        if before["prints"].get(rel) != after["prints"].get(rel):
            diffs.append("content changed: " + rel)
    return diffs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apply", action="store_true", help="move the pointers (default: dry run)")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--main", default="main")
    a = ap.parse_args(argv)
    repo, remote, main_b = a.repo, a.remote, a.main
    mode = "APPLY" if a.apply else "DRY RUN"
    print("[return_to_main] " + mode + "  repo=" + repo)

    branch = out(repo, "symbolic-ref", "--short", "-q", "HEAD", check=False)
    if not branch:
        print("REFUSED: detached HEAD; check out a named branch first.")
        return 2
    head = out(repo, "rev-parse", "HEAD")
    print("  current branch: " + branch + " @ " + head[:7])

    print("  fetching " + remote + " ...")
    git(repo, "fetch", remote)
    rmain = remote + "/" + main_b
    rsha = out(repo, "rev-parse", "-q", "--verify", rmain + "^{commit}", check=False)
    if not rsha:
        print("REFUSED: " + rmain + " does not exist after the fetch.")
        return 2
    print("  " + rmain + " @ " + rsha[:7] + "  " + out(repo, "log", "-1", "--format=%s", rsha))

    if branch == main_b:
        behind = out(repo, "rev-list", "--count", "HEAD.." + rsha)
        ahead = out(repo, "rev-list", "--count", rsha + "..HEAD")
        print("  already on " + main_b + " (" + ahead + " ahead, " + behind + " behind "
              + rmain + "); nothing to do here.")
        if behind != "0":
            print("  NOTE: catching up would change files on disk; that is a separate, "
                  "deliberate step, not this script's.")
        return 0

    lsha = out(repo, "rev-parse", "-q", "--verify", "refs/heads/" + main_b, check=False)
    if lsha:
        own = out(repo, "rev-list", "--count", rsha + ".." + lsha)
        if own != "0":
            print("REFUSED: local " + main_b + " has " + own + " commit(s) not in " + rmain
                  + "; moving it would orphan them:")
            print(out(repo, "log", "--oneline", rsha + ".." + lsha))
            return 2
        print("  local " + main_b + " @ " + lsha[:7] + " (no own commits)")
    else:
        print("  local " + main_b + " does not exist; it will be created at " + rsha[:7])

    htree = out(repo, "rev-parse", "HEAD^{tree}")
    rtree = out(repo, "rev-parse", rsha + "^{tree}")
    if htree != rtree:
        names = out(repo, "diff", "--name-status", "HEAD", rsha).splitlines()
        print("REFUSED: " + rmain + " tree != HEAD tree; switching would change "
              + str(len(names)) + " tracked path(s) on disk:")
        for line in names[:40]:
            print("    " + line)
        if len(names) > 40:
            print("    ... " + str(len(names) - 40) + " more")
        print("  (PR not merged yet, merged differently, or " + main_b
              + " moved on since; a real checkout is the user's call.)")
        return 2
    print("  tree check: " + rmain + " tree == HEAD tree (" + htree[:7]
          + "): no tracked file changes")

    anc = git(repo, "merge-base", "--is-ancestor", "HEAD", rsha, check=False).returncode == 0
    print("  branch commits on " + rmain + ": " + ("yes (merged)" if anc else
          "no (squash/rebase merge; content identical, branch keeps its own commits)"))
    rb = out(repo, "rev-parse", "-q", "--verify", remote + "/" + branch, check=False)
    print("  " + remote + "/" + branch + ": " + (rb[:7] + (" == HEAD" if rb == head else " != HEAD")
                                                 if rb else "absent"))

    before = snapshot(repo)
    print("  working tree: " + str(len(before["status_changed"])) + " modified/untracked, "
          + str(len(before["status_ignored"])) + " status lines incl. ignored")
    for xy, path in before["status_changed"]:
        print("    " + xy + " " + path + "  (stays as it is)")

    if not a.apply:
        print("DRY RUN: would run  git branch -f " + main_b + " " + rsha[:7]
              + "  then  git checkout " + main_b + "   (re-run with --apply)")
        return 0

    git(repo, "branch", "-f", main_b, rsha)
    git(repo, "checkout", main_b, "--")
    now = out(repo, "symbolic-ref", "--short", "-q", "HEAD", check=False)
    nsha = out(repo, "rev-parse", "HEAD")
    after = snapshot(repo)
    diffs = compare(before, after)
    print("  now on " + now + " @ " + nsha[:7] + "; branch " + branch + " kept @ " + head[:7])
    if now != main_b or nsha != rsha:
        print("PROBLEM: expected " + main_b + " @ " + rsha[:7])
        return 3
    if diffs:
        print("PROBLEM: the working tree differs after the switch (" + str(len(diffs)) + "):")
        for d in diffs[:60]:
            print("    " + d)
        print("  (a watcher writing at the same moment can cause this; check each path)")
        return 3
    print("  verified: status (incl. ignored) and every modified/untracked file identical")
    print("DONE")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GitError as e:
        print("GIT ERROR: " + str(e))
        sys.exit(1)
