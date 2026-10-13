"""Temp-dir test: a return never overwrites the live review-set inputs.

Reproduces the 2026-10-12 incident: the live session was re-run (new, larger
review_neuron.mat / review_summary.txt / review_report.pdf), then a return made
from the OLD bundle arrived.  copy_session must bring the review outputs across
and leave every bundle input untouched, with or without --force; a missing
input may still be created.
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ingest_returns as ir

fails = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


tmp = Path(tempfile.mkdtemp(prefix="ingest_inputs_"))
try:
    src, dst = tmp / "inbox_session", tmp / "local_session"
    src.mkdir(); dst.mkdir()
    live_inputs = {"review_neuron.mat": b"L" * 500, "review_summary.txt": b"L" * 60,
                   "review_report.pdf": b"L" * 300, "Cn.mat": b"L" * 40, "pnr.mat": b"L" * 40,
                   "Ybg_weights.mat": b"L" * 80, "ROIs_candidates.jpg": b"L" * 20,
                   "run_final_review.m": b"L" * 30}
    for name, content in live_inputs.items():
        (dst / name).write_bytes(content)
    stale_inputs = {name: b"S" * (len(content) // 2) for name, content in live_inputs.items()}
    for name, content in stale_inputs.items():
        (src / name).write_bytes(content)
    (src / "Ybg_mean.mat").write_bytes(b"NEW-INPUT")          # missing locally -> may be created
    (src / "labels.mat").write_bytes(b"LABELS")
    (src / "neuron.mat").write_bytes(b"NEURON")
    (src / "ROIs.jpg").write_bytes(b"ROIS")

    for force in (False, True):
        for name, content in live_inputs.items():
            (dst / name).write_bytes(content)
        for name in ("labels.mat", "neuron.mat", "ROIs.jpg", "Ybg_mean.mat"):
            if (dst / name).exists():
                (dst / name).unlink()
        copied, skipped, nbytes = ir.copy_session(src, dst, force=force, dry=False)
        tag = "force" if force else "default"
        check(all((dst / n).read_bytes() == c for n, c in live_inputs.items()),
              f"[{tag}] every live bundle input untouched by the stale return")
        check((dst / "labels.mat").read_bytes() == b"LABELS" and (dst / "neuron.mat").read_bytes() == b"NEURON"
              and (dst / "ROIs.jpg").read_bytes() == b"ROIS", f"[{tag}] review outputs copied")
        check((dst / "Ybg_mean.mat").read_bytes() == b"NEW-INPUT", f"[{tag}] a missing input is still created")
        check(copied == 4 and skipped == len(live_inputs), f"[{tag}] copied={copied} skipped={skipped}")

    # dry run reports nothing for the inputs either
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        ir.copy_session(src, dst, force=True, dry=True)
    check("would copy review_neuron.mat" not in buf.getvalue(), "dry run never offers to copy an input over")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
print("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): " + "; ".join(fails))
sys.exit(1 if fails else 0)
