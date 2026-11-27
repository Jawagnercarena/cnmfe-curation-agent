"""Temp-dir test for return_to_main.py.

Every repo here is made under one temp folder: a bare "origin" plus working
clones; a second clone plays GitHub and does the merges.  The real repo and its
remote are never touched (every call passes --repo <temp clone>).
"""
import io
import shutil
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import return_to_main as R

fails = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


def g(cwd, *args):
    p = subprocess.run(["git", "-C", str(cwd)] + list(args),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError("git " + " ".join(args) + ": " + p.stderr.decode("utf-8", "replace"))
    return p.stdout.decode("utf-8", "replace").strip()


def cfg(repo):
    g(repo, "config", "user.name", "test")
    g(repo, "config", "user.email", "test@example.invalid")
    g(repo, "config", "core.autocrlf", "false")


def run(*args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = R.main(list(args))
    return rc, buf.getvalue()


def write(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(text.encode("ascii"))


def fresh(tmp, name):
    """origin (bare) + work clone + gh clone, main with one commit."""
    base = tmp / name
    origin = base / "origin.git"
    g(tmp, "init", "--bare", "-b", "main", str(origin))
    work = base / "work"
    g(tmp, "clone", "-q", str(origin), str(work))
    cfg(work)
    write(work / "a.txt", "alpha\n")
    write(work / "keep.txt", "keep\n")
    write(work / ".gitignore", "*.log\nscratch/\n")
    g(work, "add", "-A")
    g(work, "commit", "-q", "-m", "init")
    g(work, "push", "-q", "origin", "main")
    gh = base / "gh"
    g(tmp, "clone", "-q", str(origin), str(gh))
    cfg(gh)
    return origin, work, gh


def branch_with_commit(work, branch="records", start="HEAD"):
    g(work, "switch", "-q", "-c", branch, start)
    write(work / "rec" / "log.txt", "record\n")
    g(work, "add", "-A")
    g(work, "commit", "-q", "-m", "records")
    g(work, "push", "-q", "-u", "origin", branch)


def gh_merge(gh, branch="records", squash=False):
    g(gh, "fetch", "-q", "origin")
    g(gh, "switch", "-q", "main")
    g(gh, "reset", "-q", "--hard", "origin/main")
    if squash:
        g(gh, "merge", "-q", "--squash", "origin/" + branch)
        g(gh, "commit", "-q", "-m", "squash")
    else:
        g(gh, "merge", "-q", "--no-ff", "-m", "Merge pull request", "origin/" + branch)
    g(gh, "push", "-q", "origin", "main")


def dirty(work):
    """Other people's uncommitted work: modified, staged, untracked, ignored."""
    write(work / "a.txt", "alpha edited by someone else\n")
    write(work / "staged.txt", "staged\n")
    g(work, "add", "staged.txt")
    write(work / "untracked.py", "x = 1\n")
    write(work / "w.log", "ignored log\n")
    write(work / "scratch" / "big.bin", "ignored dir\n")


tmp = Path(tempfile.mkdtemp(prefix="rtm_test_"))
try:
    print("1. branch pushed, PR NOT merged -> refused, nothing moves")
    origin, work, gh = fresh(tmp, "s1")
    branch_with_commit(work)
    main_before = g(work, "rev-parse", "main")
    rc, o = run("--repo", str(work), "--apply")
    check(rc == 2 and "REFUSED" in o and "rec/log.txt" in o, "exit 2, lists the differing path")
    check(g(work, "symbolic-ref", "--short", "HEAD") == "records"
          and g(work, "rev-parse", "main") == main_before, "still on records, main untouched")
    check((work / "rec" / "log.txt").exists(), "branch file still on disk")

    print("2. merged with a merge commit, dirty tree -> dry run, then apply")
    gh_merge(gh)
    dirty(work)
    keep_mtime = (work / "keep.txt").stat().st_mtime_ns
    rec_mtime = (work / "rec" / "log.txt").stat().st_mtime_ns
    rc, o = run("--repo", str(work))
    check(rc == 0 and "DRY RUN: would run" in o and "yes (merged)" in o, "dry run exit 0, says merged")
    check(g(work, "symbolic-ref", "--short", "HEAD") == "records"
          and g(work, "rev-parse", "main") == main_before, "dry run moved nothing")
    check("untracked.py  (stays as it is)" in o and "a.txt  (stays as it is)" in o,
          "dry run lists the uncommitted files")
    rc, o = run("--repo", str(work), "--apply")
    check(rc == 0 and "DONE" in o and "verified" in o, "apply exit 0, verified")
    check(g(work, "symbolic-ref", "--short", "HEAD") == "main"
          and g(work, "rev-parse", "HEAD") == g(work, "rev-parse", "origin/main"),
          "on main == origin/main")
    check((work / "a.txt").read_text() == "alpha edited by someone else\n", "modified file kept")
    check(g(work, "diff", "--cached", "--name-only") == "staged.txt", "staged file still staged")
    check((work / "untracked.py").exists() and (work / "w.log").exists()
          and (work / "scratch" / "big.bin").exists(), "untracked + ignored files kept")
    check((work / "keep.txt").stat().st_mtime_ns == keep_mtime
          and (work / "rec" / "log.txt").stat().st_mtime_ns == rec_mtime,
          "tracked files not rewritten (mtimes unchanged)")
    check(g(work, "rev-parse", "-q", "--verify", "records") != "", "branch records kept")
    rc, o = run("--repo", str(work), "--apply")
    check(rc == 0 and "already on main" in o, "second run: already on main, exit 0")

    print("3. squash merge -> tree equal, allowed, reported as squash")
    origin, work, gh = fresh(tmp, "s3")
    branch_with_commit(work)
    gh_merge(gh, squash=True)
    rc, o = run("--repo", str(work), "--apply")
    check(rc == 0 and "squash/rebase merge" in o and "DONE" in o, "exit 0, squash reported")
    check(g(work, "symbolic-ref", "--short", "HEAD") == "main", "on main")

    print("4. main moved on after the merge -> refused")
    origin, work, gh = fresh(tmp, "s4")
    branch_with_commit(work)
    gh_merge(gh)
    write(gh / "later.txt", "someone else's PR\n")
    g(gh, "add", "-A")
    g(gh, "commit", "-q", "-m", "later")
    g(gh, "push", "-q", "origin", "main")
    rc, o = run("--repo", str(work), "--apply")
    check(rc == 2 and "later.txt" in o, "exit 2, names later.txt")
    check(g(work, "symbolic-ref", "--short", "HEAD") == "records" and not (work / "later.txt").exists(),
          "still on records, no file added")

    print("5. local main has its own commit -> refused")
    origin, work, gh = fresh(tmp, "s5")
    write(work / "local_only.txt", "x\n")
    g(work, "add", "-A")
    g(work, "commit", "-q", "-m", "local-only commit on main")
    lm = g(work, "rev-parse", "main")
    branch_with_commit(work, start="origin/main")
    gh_merge(gh)
    rc, o = run("--repo", str(work), "--apply")
    check(rc == 2 and "local-only commit on main" in o, "exit 2, shows the orphan commit")
    check(g(work, "rev-parse", "main") == lm, "local main not moved")

    print("6. detached HEAD -> refused")
    g(work, "checkout", "-q", "--detach")
    rc, o = run("--repo", str(work), "--apply")
    check(rc == 2 and "detached" in o, "exit 2")

    print("7. no local main branch -> created at origin/main")
    origin, work, gh = fresh(tmp, "s7")
    branch_with_commit(work)
    gh_merge(gh)
    g(work, "branch", "-q", "-D", "main")
    rc, o = run("--repo", str(work), "--apply")
    check(rc == 0 and "will be created" in o and g(work, "symbolic-ref", "--short", "HEAD") == "main",
          "exit 0, main created and checked out")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\nRESULT: " + ("ALL PASS" if not fails else str(len(fails)) + " FAILURE(S)"))
sys.exit(1 if fails else 0)
