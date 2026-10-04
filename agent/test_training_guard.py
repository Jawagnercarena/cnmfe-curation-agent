"""Temp-dir test: ingest_returns never imports a training sandbox.

A trainee who copies a training folder (marked TRAINING_SESSION.txt) into
inbox/ must not be able to overwrite the real session: iter_sessions skips the
folder (and anything under a marked ancestor), and the explicit-path branch in
main() uses the same check.
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ingest_returns as ir
import training_common as tc

fails = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


tmp = Path(tempfile.mkdtemp(prefix="training_guard_"))
try:
    check(ir.TRAINING_MARKER == tc.TRAINING_MARKER == "TRAINING_SESSION.txt",
          "marker name agrees between ingest_returns and training_common")
    inbox = tmp / "inbox"
    s1 = inbox / "Alice" / "BLA" / "t" / "s1"; s1.mkdir(parents=True); (s1 / "labels.mat").write_bytes(b"x")
    s2 = inbox / "Alice" / "BLA" / "t" / "s2"; s2.mkdir(parents=True); (s2 / "labels.mat").write_bytes(b"x")
    (s2 / ir.TRAINING_MARKER).write_text("trainee: Alice\n", encoding="ascii")
    s3 = inbox / "Bob" / "training_stuff" / "BLA" / "t" / "s3"; s3.mkdir(parents=True); (s3 / "neuron.mat").write_bytes(b"x")
    (inbox / "Bob" / "training_stuff" / ir.TRAINING_MARKER).write_text("trainee: Bob\n", encoding="ascii")
    s4 = inbox / "Bob" / ".parked" / "s4"; s4.mkdir(parents=True); (s4 / "labels.mat").write_bytes(b"x")
    s5 = inbox / "Bob" / "vCA1" / "t" / "s5"; s5.mkdir(parents=True); (s5 / "labels.mat").write_bytes(b"x")

    got = {p.name for p in ir.iter_sessions(inbox)}
    check(got == {"s1", "s5"}, f"iter_sessions yields only unmarked sessions (got {sorted(got)})")
    check(ir.is_training_sandbox(s2, inbox), "marker on the session folder -> sandbox")
    check(ir.is_training_sandbox(s3, inbox), "marker on an ancestor -> sandbox")
    check(not ir.is_training_sandbox(s1, inbox), "no marker -> not a sandbox")
    check(not ir.is_training_sandbox(s5, inbox), "unrelated reviewer folder -> not a sandbox")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\nRESULT: {'ALL PASS' if not fails else str(len(fails)) + ' FAILURE(S)'}")
sys.exit(1 if fails else 0)
