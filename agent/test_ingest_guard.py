"""Temp-dir test for the CENTRAL_ONLY ingest guard (Step 0).

Simulates a reviewer who mirrored whole central folders back into their inbox:
the returned bundle carries a STALE 13-column candidate_features.npz and a stale
labels_provenance.txt alongside the real labels.mat.  The local session is on the
35-column contract.  copy_session must bring the labels across and leave both
central-only files untouched.
"""
import shutil, sys, tempfile
from pathlib import Path

sys.path.insert(0, r"c:\code\CNMF_E_LEGACY_BIANE_CLAUDE\agent")
import numpy as np
import ingest_returns as ir

fails = []
def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)

tmp = Path(tempfile.mkdtemp(prefix="ingest_guard_"))
try:
    src, dst = tmp / "inbox_session", tmp / "local_session"
    src.mkdir(); dst.mkdir()

    # local (central) state: 35-column contract + a real provenance record
    np.savez(dst / "candidate_features.npz",
             feature_matrix=np.zeros((40, 35)), feature_names=np.array(["c"] * 35),
             auto_rejected=np.array([1, 2], dtype=int), n_candidates=np.array([40]))
    (dst / "labels_provenance.txt").write_text("reviewer: Taylor\n", encoding="utf-8")
    local_npz_bytes = (dst / "candidate_features.npz").read_bytes()
    local_prov = (dst / "labels_provenance.txt").read_text(encoding="utf-8")

    # returned bundle: real labels + STALE mirrored central files (13-col npz)
    np.savez(src / "candidate_features.npz",
             feature_matrix=np.zeros((40, 13)), feature_names=np.array(["c"] * 13),
             auto_rejected=np.array([], dtype=int), n_candidates=np.array([40]))
    (src / "labels_provenance.txt").write_text("reviewer: STALE\n", encoding="utf-8")
    (src / "labels.mat").write_bytes(b"LABELS-FROM-REVIEWER")
    (src / "review_neuron.mat").write_bytes(b"REVIEW-NEURON")
    (src / ".gsig9_wrongparams").mkdir()
    (src / ".gsig9_wrongparams" / "neuron.mat").write_bytes(b"PARKED")

    sizes_differ = (src / "candidate_features.npz").stat().st_size != len(local_npz_bytes)
    check(sizes_differ, "the stale npz differs in size (so the same-size skip would NOT have caught it)")

    copied, skipped, nbytes = ir.copy_session(src, dst, force=False, dry=False)
    print(f"  copy_session -> copied={copied} skipped={skipped} bytes={nbytes}")

    check((dst / "candidate_features.npz").read_bytes() == local_npz_bytes,
          "local candidate_features.npz untouched (still 35-col)")
    check(np.load(dst / "candidate_features.npz", allow_pickle=True)["feature_matrix"].shape[1] == 35,
          "local feature width still 35")
    check((dst / "labels_provenance.txt").read_text(encoding="utf-8") == local_prov,
          "local labels_provenance.txt untouched")
    check((dst / "labels.mat").read_bytes() == b"LABELS-FROM-REVIEWER",
          "labels.mat WAS copied (the guard does not block real returns)")
    check((dst / "review_neuron.mat").read_bytes() == b"REVIEW-NEURON",
          "review_neuron.mat WAS copied")
    check(not (dst / ".gsig9_wrongparams").exists(),
          "dot-prefixed parked tree still skipped")

    # --force must not override the guard either
    copied2, _, _ = ir.copy_session(src, dst, force=True, dry=False)
    check((dst / "candidate_features.npz").read_bytes() == local_npz_bytes,
          "--force does NOT override the guard")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\nRESULT: {'ALL PASS' if not fails else str(len(fails)) + ' FAILURE(S)'}")
sys.exit(1 if fails else 0)
