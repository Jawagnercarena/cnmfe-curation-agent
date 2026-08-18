"""Pool-manifest helpers: pin the labeled-session pool so every experiment in
this investigation sees the identical corpus, and fail loudly if a reviewer
return lands mid-analysis."""
import json
import sys
from pathlib import Path

AGENT = Path(r"c:\code\CNMF_E_LEGACY_BIANE_CLAUDE\agent")
if str(AGENT) not in sys.path:
    sys.path.insert(0, str(AGENT))

MANIFEST = Path(__file__).parent / "pool_manifest.json"


def build_state():
    import numpy as np
    import train_classifier as _tc
    from config import DATA_ROOT
    state = []
    for td in sorted(DATA_ROOT.iterdir()):
        if not td.is_dir() or td.name.startswith("."):
            continue
        for sd in sorted(td.iterdir()):
            if not sd.is_dir():
                continue
            npz_f, lab_f = sd / "candidate_features.npz", sd / "labels.mat"
            if not npz_f.exists() or not lab_f.exists():
                continue
            npz = np.load(npz_f, allow_pickle=True)
            state.append({
                "rel": f"{td.name}/{sd.name}",
                "is_bootstrap": bool(_tc._is_bootstrap_session(sd)),
                "n_candidates": int(npz["n_candidates"][0]),
                "labels_mtime": lab_f.stat().st_mtime,
            })
    return state


def write_manifest():
    state = build_state()
    MANIFEST.write_text(json.dumps(state, indent=1))
    n_ag = sum(1 for s in state if not s["is_bootstrap"])
    print(f"manifest written: {len(state)} sessions ({n_ag} agent, "
          f"{len(state) - n_ag} bootstrap) -> {MANIFEST}")
    return state


def assert_unchanged():
    """Compare the live pool against the pinned manifest; raise on drift."""
    pinned = json.loads(MANIFEST.read_text())
    live = build_state()
    p = {s["rel"]: s for s in pinned}
    l = {s["rel"]: s for s in live}
    drift = []
    for rel in p.keys() | l.keys():
        if rel not in p:
            drift.append(f"NEW session since manifest: {rel}")
        elif rel not in l:
            drift.append(f"session GONE since manifest: {rel}")
        elif (p[rel]["labels_mtime"] != l[rel]["labels_mtime"]
              or p[rel]["n_candidates"] != l[rel]["n_candidates"]):
            drift.append(f"session CHANGED since manifest: {rel}")
    if drift:
        raise RuntimeError("POOL DRIFT — numbers are not comparable:\n  "
                           + "\n  ".join(drift))
    return pinned
