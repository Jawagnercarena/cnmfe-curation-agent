"""
Step 4 pool re-pin.  Writes step4_pool_manifest.json (current labeled pool,
same schema as the Step 2 pin) and reports drift against the Step 2 pinned
manifest — the pool has been live since 2026-08-18, so new/changed sessions
are expected and each one is listed so the backfill can handle it explicitly.

Also reports the pending (curated, unlabeled) sessions the backfill must
upgrade, and provenance-checks any labeled session that is NEW since the pin
(review_neuron.mat must predate labels.mat; extraction row counts must match)
before it is allowed into the v2b extraction list.
"""
import json
import sys
from pathlib import Path

import numpy as np

SP = Path(__file__).parent
AGENT = Path(r"c:\code\CNMF_E_LEGACY_BIANE_CLAUDE\agent")
sys.path.insert(0, str(AGENT))
sys.path.insert(0, str(AGENT / "eval" / "step2_2026-08"))

import manifest_util  # step2 helper; build_state() reads the live pool

STEP2_MANIFEST = Path(r"D:\Julian_CNMFe\BLA\.feature_expansion\_pinned\pool_manifest.json")
OUT = SP / "step4_pool_manifest.json"
EXT = Path(r"D:\Julian_CNMFe\BLA\.feature_expansion")
DATA_ROOT = Path(r"D:\Julian_CNMFe\BLA")


def find_pending():
    out = []
    for td in sorted(DATA_ROOT.iterdir()):
        if not td.is_dir() or td.name.startswith("."):
            continue
        for sd in sorted(td.iterdir()):
            if not sd.is_dir():
                continue
            if ((sd / "ROIs_candidates.jpg").exists()
                    and (sd / "candidate_features.npz").exists()
                    and not (sd / "labels.mat").exists()):
                out.append(sd)
    return out


def main():
    live = manifest_util.build_state()
    OUT.write_text(json.dumps(live, indent=1))
    n_ag = sum(1 for s in live if not s["is_bootstrap"])
    print(f"step4 manifest written: {len(live)} labeled sessions "
          f"({n_ag} agent, {len(live) - n_ag} bootstrap) -> {OUT.name}")

    pinned = {s["rel"]: s for s in json.loads(STEP2_MANIFEST.read_text())}
    livem = {s["rel"]: s for s in live}
    new = sorted(set(livem) - set(pinned))
    gone = sorted(set(pinned) - set(livem))
    changed = sorted(r for r in set(pinned) & set(livem)
                     if pinned[r]["labels_mtime"] != livem[r]["labels_mtime"]
                     or pinned[r]["n_candidates"] != livem[r]["n_candidates"])
    print(f"\nDrift vs the Step 2 pin (170 sessions):")
    print(f"  new: {len(new)}  gone: {len(gone)}  changed: {len(changed)}")
    for r in new:
        print(f"    NEW     {r}")
    for r in gone:
        print(f"    GONE    {r}")
    for r in changed:
        print(f"    CHANGED {r}")

    # Provenance gate for labeled agent sessions missing a v2b extraction
    need_extract = []
    bad = []
    for s in live:
        if s["is_bootstrap"]:
            continue
        rel = s["rel"]
        if (EXT / (rel.replace("/", "__") + ".mat")).exists():
            continue
        sd = DATA_ROOT / rel
        rn, lab = sd / "review_neuron.mat", sd / "labels.mat"
        if not rn.exists():
            bad.append((rel, "no review_neuron.mat"))
            continue
        if rn.stat().st_mtime >= lab.stat().st_mtime:
            bad.append((rel, "review_neuron.mat NOT older than labels.mat"))
            continue
        npz = np.load(sd / "candidate_features.npz", allow_pickle=True)
        n_rev = int(npz["n_candidates"][0]) - len(npz["auto_rejected"])
        import scipy.io as sio
        n_lab = len(sio.loadmat(str(lab))["labels"].flatten())
        if n_rev != n_lab:
            bad.append((rel, f"review set {n_rev} != labels {n_lab}"))
            continue
        need_extract.append(rel)
    print(f"\nLabeled agent sessions needing v2b extraction: {len(need_extract)}")
    for r in need_extract:
        print(f"    EXTRACT {r}")
    if bad:
        print(f"\nPROVENANCE FAILURES ({len(bad)}) — STOP, do not extract these:")
        for r, why in bad:
            print(f"    {r}: {why}")

    pend = find_pending()
    print(f"\nPending (curated, unlabeled) sessions to upgrade: {len(pend)}")
    for sd in pend:
        have = all((sd / f).exists()
                   for f in ("C_raw.txt", "spatial_footprints.mat"))
        print(f"    PENDING {sd.parent.name}/{sd.name}"
              f"{'' if have else '   [MISSING candidate files!]'}")

    (SP / "step4_extract_sessions.txt").write_text(
        "\n".join(need_extract) + ("\n" if need_extract else ""))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
