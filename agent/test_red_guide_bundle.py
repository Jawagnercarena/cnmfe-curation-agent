"""Temp-dir test for the red guide in review bundles (push_review_bundle.py) and
its skip rule on the way back (ingest_returns.py).

Everything lives under one temp folder: a fake data tree, a fake retro results
tree and a fake exchange.  RETRO_RESULTS is pointed at the temp tree before any
lookup, so the real retro project and the real exchange are never touched.
subprocess.run is replaced for the whole test, so the retro guide hook never
runs; each case sets the exit code and output it returns.
"""
import io
import shutil
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import push_review_bundle as P
import ingest_returns as ir
import rereview as rr

fails = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


def quiet(fn, *a, **kw):
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = fn(*a, **kw)
    return res, buf.getvalue()


def make_session(task_dir: Path, name: str, n: int, auto):
    sd = task_dir / name
    sd.mkdir(parents=True)
    np.savez(sd / "candidate_features.npz", feature_matrix=np.zeros((n, 13)),
             feature_names=np.array([f"f{i}" for i in range(13)]),
             auto_rejected=np.array(auto, dtype=int), n_candidates=np.array([n]))
    for fname, size in [("review_neuron.mat", 3000), ("Cn.mat", 200), ("pnr.mat", 100),
                        ("Ybg_weights.mat", 500), ("review_report.pdf", 400),
                        ("review_summary.txt", 50), ("ROIs_candidates.jpg", 80),
                        (f"{name}.mat", 10000)]:
        (sd / fname).write_bytes(b"x" * size)
    return sd


def make_guide(root: Path, animal: str, date6: str, name: str, n: int, auto, png=b"PNG-GUIDE",
               shown=True):
    g = root / animal / date6 / name / "red_guide"
    g.mkdir(parents=True)
    (g / "red_guide.png").write_bytes(png)
    if shown:
        review = [i for i in range(n) if i not in set(auto)]
        np.savez(g / "candidates_shown.npz", review_idx=np.array(review, dtype=np.int64),
                 auto_rejected=np.array(auto, dtype=np.int64), n_candidates=np.array([n]))
    return g / "red_guide.png"


real_run = subprocess.run
hook = {"calls": [], "result": (0, "red guide ok: current: x\n", ""), "exc": None}


def fake_run(cmd, **kw):
    hook["calls"].append((cmd, kw))
    if hook["exc"] is not None:
        raise hook["exc"]
    code, out, err = hook["result"]
    return subprocess.CompletedProcess(cmd, code, out, err)


subprocess.run = fake_run
tmp = Path(tempfile.mkdtemp(prefix="red_guide_bundle_"))
try:
    data = tmp / "data"
    task = data / "vCA1" / "6odorDualDiffRew"
    retro = tmp / "retro_results"
    retro.mkdir()
    exchange = tmp / "exchange"
    (exchange / "outbox").mkdir(parents=True)
    P.DATA_PARENT = data
    P.RETRO_RESULTS = str(retro)          # never the real retro tree

    ok_name = "AVG5x-TSeries-090526-pnb99-330um-3z-000"
    sd = make_session(task, ok_name, 10, [1, 2])
    src_png = make_guide(retro, "pnb99", "090526", ok_name, 10, [1, 2])
    # a parked copy in a dot folder must never count as a match
    dot = retro / "pnb99" / ".old_2026-11-02" / ok_name / "red_guide"
    dot.mkdir(parents=True)
    (dot / "red_guide.png").write_bytes(b"OLD")

    print("matching guide")
    files, out = quiet(P.collect_files, sd)
    check(src_png in files, "matching guide is in the bundle list")
    check("red guide: included" in out, "push prints 'red guide: included'")
    check(sum(1 for f in files if f.name == "red_guide.png") == 1, "dot-folder copy ignored (one guide)")
    _, out = quiet(P.stage_one, sd, str(exchange), True, "Taylor")
    dest = exchange / "outbox" / "Taylor" / "vCA1" / "6odorDualDiffRew" / ok_name
    check("would copy red_guide.png" in out and not dest.exists(), "dry run lists the guide, writes nothing")
    _, out = quiet(P.stage_one, sd, str(exchange), False, "Taylor")
    check((dest / "red_guide.png").read_bytes() == b"PNG-GUIDE", "guide copied into the outbox bundle")
    check((dest / "review_neuron.mat").exists() and (dest / "run_final_review.m").exists(),
          "rest of the bundle staged as before")
    check(not (sd / "red_guide.png").exists(), "nothing written into the session folder")
    check(src_png.read_bytes() == b"PNG-GUIDE" and (src_png.parent / "candidates_shown.npz").exists(),
          "retro tree untouched")

    print("in-place re-push compares contents, not size")
    src_png.write_bytes(b"PNG-GUIDF")                               # redrawn, same size
    (sd / "Cn.mat").write_bytes(b"y" * 200)                         # re-saved, same size
    _, out = quiet(P.stage_one, sd, str(exchange), False, "Taylor")
    check((dest / "red_guide.png").read_bytes() == b"PNG-GUIDF"
          and "copy red_guide.png (0.0 MB) -- replaces a different copy" in out,
          "same-size regenerated guide replaces the outbox copy")
    check((dest / "Cn.mat").read_bytes() == b"y" * 200, "same-size changed Cn.mat replaces the outbox copy")
    check("skip (already staged): review_neuron.mat" in out and f"skip (already staged): {ok_name}.mat" in out,
          "identical files are still skipped")
    src_png.write_bytes(b"PNG-GUIDE")
    (sd / "Cn.mat").write_bytes(b"x" * 200)
    check(P.same_contents(sd / "Cn.mat", sd / "Cn.mat") and not P.same_contents(sd / "Cn.mat", dest / "nope"),
          "same_contents: identical file True, missing target False")

    print("stale guides")
    st_name = "AVG5x-TSeries-090626-pnb99-459um-3z-000"
    sd_st = make_session(task, st_name, 76, [3, 4])
    make_guide(retro, "pnb99", "090626", st_name, 30, [3])          # drawn before a re-run
    files, out = quiet(P.collect_files, sd_st)
    check(not any(f.name == "red_guide.png" for f in files), "stale guide (N differs) NOT in the bundle")
    check("STALE, NOT included" in out and "30 candidates / 1 auto-rejected" in out
          and "76 / 2" in out, "push says why: drawn for 30/1, live 76/2")

    sa_name = "AVG5x-TSeries-090726-pnb99-463um-3z-000"
    sd_sa = make_session(task, sa_name, 10, [1, 2])
    make_guide(retro, "pnb99", "090726", sa_name, 10, [1, 5])       # recurate changed the set
    files, out = quiet(P.collect_files, sd_sa)
    check(not any(f.name == "red_guide.png" for f in files) and "different auto-rejected" in out,
          "same counts, different auto-rejected set -> STALE")

    ns_name = "AVG5x-TSeries-090826-pnb99-463um-3z-000"
    sd_ns = make_session(task, ns_name, 10, [1, 2])
    make_guide(retro, "pnb99", "090826", ns_name, 10, [1, 2], shown=False)
    files, out = quiet(P.collect_files, sd_ns)
    check(not any(f.name == "red_guide.png" for f in files) and "cannot check" in out,
          "no candidates_shown.npz -> not included (cannot check)")

    print("lookup rules")
    nf_name = "AVG5x-TSeries-090926-pnb99-463um-3z-000"
    sd_nf = make_session(task, nf_name, 5, [])
    files, out = quiet(P.collect_files, sd_nf)
    check(not any(f.name == "red_guide.png" for f in files)
          and "red guide: not found (run retro guide-for-push first)" in out, "pnb session without a guide")

    bla_name = "AVG5x-TSeries-090526-bla12-300um-3z-000"
    sd_bla = make_session(data / "BLA" / "task1", bla_name, 5, [])
    files, out = quiet(P.collect_files, sd_bla)
    check("red guide" not in out and not any(f.name == "red_guide.png" for f in files),
          "non-retro session: no guide, no message")

    d8_name = "AVG5x-TSeries-07152026-pnb98-471um-3z-000"            # 8-digit date in the name
    sd_d8 = make_session(data / "vCA1" / "3odor", d8_name, 4, [0])
    make_guide(retro, "pnb98", "071526", d8_name, 4, [0])           # 6-digit date folder
    files, out = quiet(P.collect_files, sd_d8)
    check(any(f.name == "red_guide.png" for f in files), "matched by session name, not a rebuilt path")

    dup_name = "AVG5x-TSeries-091026-pnb104-485um-3z-000"
    sd_dup = make_session(task, dup_name, 4, [])
    make_guide(retro, "pnb104", "091026", dup_name, 4, [])
    make_guide(retro, "pnb104x", "091026", dup_name, 4, [])
    try:
        quiet(P.collect_files, sd_dup)
        check(False, "two matching guides must raise")
    except ValueError as e:
        check("2 red guides match" in str(e), "two matching guides -> ValueError, nothing picked")

    print("push-time guide hook (subprocess mocked)")
    hook["result"] = (0, "making the red guide\nred guide ok: current: x\n", "")
    files, out = quiet(P.collect_files, sd, True, True)
    cmd, kw = hook["calls"][-1]
    check(cmd[:5] == [P.PYTHON_EXE, "-m", "retro_align", "guide-for-push", str(sd)]
          and cmd[-1] == "--dry-run" and Path(kw["cwd"]) == retro.parent
          and kw["timeout"] == P.GUIDE_HOOK_TIMEOUT,
          "hook command, --dry-run passed through, cwd = retro project, timeout set")
    check(src_png in files and "red guide ok: current: x" in out and "red guide: included" in out,
          "exit 0 -> the lookup adds the guide")
    files, out = quiet(P.collect_files, sd, False, True)
    check("--dry-run" not in hook["calls"][-1][0] and src_png in files, "real push: no --dry-run")
    base = [f for f in files if f.name != "red_guide.png"]

    hook["result"] = (1, "red guide NOT MADE: no usable single images\n", "")
    files, out = quiet(P.collect_files, sd, False, True)
    check(files == base and "red guide NOT MADE: no usable single images" in out,
          "exit 1 -> no guide although one on disk passes the lookup; rest of the list unchanged")
    hook["result"] = (3, "red guide would be made: would be made (no red_guide.png)\n", "")
    files, out = quiet(P.collect_files, sd, True, True)
    check(files == base and "red guide: would be built at push (no red_guide.png)" in out
          and "not found" not in out and "STALE" not in out,
          "exit 3 (dry run) -> 'would be built at push' instead of the lookup messages")
    hook["result"] = (2, "red guide not applicable: not a session folder under vCA1\n", "")
    files, out = quiet(P.collect_files, sd, False, True)
    check(files == base and "red guide not applicable" in out, "exit 2 -> no guide")
    hook["exc"] = subprocess.TimeoutExpired(["hook"], P.GUIDE_HOOK_TIMEOUT)
    files, out = quiet(P.collect_files, sd, False, True)
    check(files == base and "timed out" in out, "timeout -> pushed without a guide")
    hook["exc"] = FileNotFoundError(2, "No such file or directory", "python.exe")
    files, out = quiet(P.collect_files, sd, False, True)
    check(files == base and "could not start (FileNotFoundError" in out, "OSError -> pushed without a guide")
    hook["exc"] = None
    hook["result"] = (1, "fetching\n", "Traceback\nModuleNotFoundError: No module named 'retro_align'\n")
    files, out = quiet(P.collect_files, sd, False, True)
    check(files == base and "ModuleNotFoundError" in out and "pushing without a guide" in out,
          "crash without a result line -> stderr's last line printed, no guide")
    hook["result"] = (0, "something else\n", "")
    files, out = quiet(P.collect_files, sd, False, True)
    check(files == base, "exit 0 without the result line is not trusted")
    hook["result"] = (1, "red guide NOT MADE: caf\u00e9 \u2192 x\n", "")
    files, out = quiet(P.collect_files, sd, False, True)
    check(all(ord(c) < 128 for c in out) and "red guide NOT MADE: caf? ? x" in out,
          "hook output is printed as ASCII")
    hook["result"] = (0, "red guide ok: made: x\n", "")
    files, out = quiet(P.collect_files, sd_st, False, True)
    check(not any(f.name == "red_guide.png" for f in files) and "STALE, NOT included" in out,
          "exit 0 but the lookup finds it stale -> not included (the lookup stays the gate)")

    hook["result"] = (1, "red guide NOT MADE: X: unreachable\n", "")
    _, out = quiet(P.stage_one, sd, str(exchange), False, "Aneesh")
    d2 = exchange / "outbox" / "Aneesh" / "vCA1" / "6odorDualDiffRew" / ok_name
    check(not (d2 / "red_guide.png").exists() and (d2 / "review_neuron.mat").exists()
          and (d2 / "run_final_review.m").exists() and "staged" in out,
          "failed hook: the session is still staged, without a guide")

    n_calls = len(hook["calls"])
    P.RETRO_GUIDE_BUILD = False
    files, out = quiet(P.collect_files, sd, False, True)
    check(len(hook["calls"]) == n_calls and src_png in files,
          "off switch: no hook call, an existing matching guide is still added")
    P.RETRO_GUIDE_BUILD = True
    files, out = quiet(P.collect_files, sd)
    check(len(hook["calls"]) == n_calls and src_png in files,
          "default collect_files (rereview.py): no hook call, lookup only")
    quiet(P.collect_files, sd_bla, False, True)
    check(len(hook["calls"]) == n_calls, "non-retro session: no hook call")
    hook["result"] = (0, "red guide ok: current: x\n", "")

    P.RETRO_RESULTS = ""
    quiet(P.collect_files, sd, False, True)
    check(len(hook["calls"]) == n_calls, "empty RETRO_RESULTS: no hook call either")
    files, out = quiet(P.collect_files, sd)
    check(not any(f.name == "red_guide.png" for f in files) and "root '' missing" in out,
          "empty RETRO_RESULTS = feature off")
    P.RETRO_RESULTS = str(tmp / "no_such_root")
    files, out = quiet(P.collect_files, sd)
    check(not any(f.name == "red_guide.png" for f in files), "missing retro root = feature off")
    P.RETRO_RESULTS = str(retro)

    check("red_guide.png" not in rr.REVIEW_OUTPUT_FILES,
          "rereview's blind-bundle leak check does not reject the guide")

    print("ingest")
    ret = tmp / "inbox_return" / ok_name
    ret.mkdir(parents=True)
    (ret / "labels.mat").write_bytes(b"LABELS")
    (ret / "red_guide.png").write_bytes(b"PNG-FROM-REVIEWER")
    dst = tmp / "local_session"
    dst.mkdir()
    (c, s, b), _ = quiet(ir.copy_session, ret, dst, False, False)
    check((dst / "labels.mat").read_bytes() == b"LABELS", "labels.mat copied")
    check(not (dst / "red_guide.png").exists(), "returned red_guide.png NOT copied into the session")
    quiet(ir.copy_session, ret, dst, True, False)
    check(not (dst / "red_guide.png").exists(), "--force does not copy it either")
    _, out = quiet(ir.print_summary_table,
                   [("Taylor", "vCA1/x/" + ok_name, "NEW", 1, 6, "1/2/0", "-", "yes"),
                    ("Taylor", "vCA1/x/other", "SKIPPED", 0, 0, "-")], False)
    check("red guide in return" in out and " yes" in out, "summary table has the red guide column")
finally:
    subprocess.run = real_run
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\nRESULT: {'ALL PASS' if not fails else str(len(fails)) + ' FAILURE(S)'}")
sys.exit(1 if fails else 0)
