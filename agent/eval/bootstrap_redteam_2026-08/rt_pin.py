"""
rt_pin.py -- pin the corpus the red team measures.

    python rt_pin.py            write pin_manifest.json (refuses to overwrite
                                an existing pin unless --force)
    python rt_pin.py --check    recompute and diff against the stored pin;
                                exit 1 on any drift

The pin records, per area (BLA, vCA1, DG_AL): every session with a
candidate_features.npz, whether it is labeled / bootstrap / pending, the
labels.mat mtime + size + sha256, the sha256 of bootstrap_match_stats.json,
the npz size + mtime, and the deployed joblib's md5.  pin_hash is the sha256
of that table; every results JSON carries it.  Read-only.
"""
import argparse
import json
import sys
from pathlib import Path

import rt_lib as rt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    new = rt.compute_pin()
    for area, a in new["areas"].items():
        print(f"{area}: labeled {a['n_labeled']} (agent {a['n_agent']}, bootstrap "
              f"{a['n_bootstrap']}), pending {a['n_pending']}, joblib md5 {a['joblib']['md5']}")
    print(f"pin_hash {new['pin_hash']}")

    if args.check:
        old = rt.load_pin()
        d = rt.diff_pin(old, new)
        drift = False
        for area, v in d.items():
            n = len(v["new"]) + len(v["gone"]) + len(v["changed"])
            print(f"{area}: new {len(v['new'])} / gone {len(v['gone'])} / changed "
                  f"{len(v['changed'])} / joblib changed {v['joblib_changed']}")
            for k in ("new", "gone", "changed"):
                for s in v[k]:
                    print(f"    {k}: {s}")
            drift |= bool(n) or v["joblib_changed"]
        print("PIN CHECK:", "DRIFT" if drift else "unchanged",
              f"(stored {old['pin_hash'][:12]}, now {new['pin_hash'][:12]})")
        return 1 if drift else 0

    if rt.PIN_FILE.exists() and not args.force:
        print(f"{rt.PIN_FILE.name} exists -- use --check to compare or --force to re-pin")
        return 2
    rt.PIN_FILE.write_text(json.dumps(new, indent=1))
    print(f"wrote {rt.PIN_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
