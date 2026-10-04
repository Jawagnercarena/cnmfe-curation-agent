"""
prep_animal_map.py -> results/animal_map.json

Animal / date / FOV for every session with a candidate_features.npz in the
three areas (labeled + pending), plus the manual-check table the brief asks
for (names that resolve to zero or several animal ids).  FOV proxy:
  BLA / vCA1 : (animal, first depth token 'NNNum' after the animal)
  DG_AL      : (date, animal, depth tokens) -- A/B planes of one day, and the
               non-averaged TSeries duplicate of an AVG4x recording, are one FOV.
Also reports the animal overlap between agent CV sessions and bootstrap
sessions per area (attack #8's leak channel).  Read-only.
"""
import re
import sys
from collections import Counter, defaultdict

import rt_lib as rt


def depth_tokens(name):
    toks = rt._tokens_after_tseries(name)
    return [t for t in toks if re.fullmatch(r"\d+um", t)]


def fov_of(area, name, animal, date):
    d = depth_tokens(name)
    if area == "DG_AL":
        return f"{date}|{animal}|{'-'.join(d)}"
    return f"{animal}|{d[0] if d else '?'}"


def main():
    t = rt.Timer()
    rt.assert_pinned()
    keys = rt.animal_keys()
    out = {"areas": {}, "manual_check": []}
    for area in rt.AREAS:
        rows = []
        for sd in rt.iter_sessions(area):
            if not (sd / "candidate_features.npz").exists():
                continue
            labeled = (sd / "labels.mat").exists()
            animal = rt.animal_of(area, sd.name)
            date = rt.date_of(sd.name)
            toks = rt._tokens_after_tseries(sd.name)
            numeric_hits = [x for x in toks[1:] if x.isdigit() and x in keys]
            named_hits = re.findall(r"-(bla\d+|pnb\d+|DG6[A-Z])-", sd.name)
            n_pos = n_rows = None
            is_bs = None
            if labeled:
                r = rt.load_record(sd, area)
                n_pos, n_rows, is_bs = int(r["y"].sum()), int(len(r["y"])), bool(r["is_bootstrap"])
            row = {"rel": rt.rel(sd), "labeled": labeled, "is_bootstrap": is_bs,
                   "animal": animal, "date": date.isoformat() if date else None,
                   "fov": fov_of(area, sd.name, animal, date),
                   "n_pos": n_pos, "n_rows": n_rows,
                   "cv_eligible": bool(labeled and not is_bs and n_pos is not None and n_pos >= rt.MIN_POS)}
            rows.append(row)
            if animal is None or len(numeric_hits) + len(named_hits) != 1:
                out["manual_check"].append({"area": area, "rel": rt.rel(sd), "resolved": animal,
                                            "numeric_hits": numeric_hits, "named_hits": named_hits})
        cv_animals = Counter(r["animal"] for r in rows if r["cv_eligible"])
        bs_animals = Counter(r["animal"] for r in rows if r["is_bootstrap"])
        overlap = {a: {"cv_sessions": cv_animals[a], "bootstrap_sessions": bs_animals[a]}
                   for a in cv_animals if a in bs_animals}
        fovs = defaultdict(list)
        for r in rows:
            if r["labeled"]:
                fovs[r["fov"]].append(r["rel"])
        shared_fov = {k: v for k, v in fovs.items() if len(v) > 1}
        out["areas"][area] = {
            "sessions": rows, "n": len(rows), "n_labeled": sum(r["labeled"] for r in rows),
            "cv_animals": dict(cv_animals), "bootstrap_animals": dict(bs_animals),
            "agent_bootstrap_animal_overlap": overlap,
            "n_fov_groups_labeled": len(fovs), "fovs_with_multiple_labeled_sessions": shared_fov,
        }
        rt.log(f"{area}: {len(rows)} sessions, CV animals {dict(cv_animals)}, bootstrap animals "
               f"{len(bs_animals)}, overlap {overlap}, labeled FOV groups {len(fovs)} "
               f"({len(shared_fov)} with >1 labeled session)")
    rt.log(f"manual-check rows: {len(out['manual_check'])}")
    for m in out["manual_check"]:
        rt.log(f"   {m}")
    rt.write_result("animal_map", out, __file__, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
