"""C10 (section 4 of redteam/c10_minpnr.py) on every labelled cohort-5 6odor session: does a
higher min_pnr separate kept from non-kept candidates?  Read-only.

For every candidate that can seed from its own core at the run's thresholds (ceiling at the
run's min_pnr >= the run's min_corr): can it still do so at min_pnr + d?  'lost' = no.
This is the same ceiling ESTIMATE as in c8_c9_batch.py (first seeding image only), not a re-run.
Categories: kept | reviewed delete (dup-of-kept / other) | auto-rejected at 0.01 (never
shown).  The 11-11 result (12 threshold-0 pilots) pooled every non-kept candidate as
'rejected'; that pooled figure is printed too so the two can be compared.
A lost kept cell is 'covered' when another candidate overlapping it (support IoU >= 0.5)
still seeds at min_pnr + d, i.e. the cell would probably still be found once.
Usage: c10_minpnr_batch.py > c10_minpnr_batch.txt
"""
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sd_lib as sl  # noqa: E402

DPS = (0, 1, 2, 3, 4, 6)
CATS = ("kept", "del_dup", "del_other", "auto")


def pct(a, b):
    return "%.1f%%" % (100.0 * a / b) if b else "-"


def main():
    S = sl.sessions()
    print("labelled sessions: %d" % len(S))
    res = {}          # group -> dp -> cat -> [lost, base]
    cov = {}          # group -> dp -> kept lost but covered
    per = []
    marg = {}         # group -> (margin, full)
    for s in S:
        cn, pnr, mc, mp, full = s["cn"], s["pnr"], s["mc"], s["mp"], s["full"]
        cat = np.where(full == 1, 0, np.where(s["dup"], 1, np.where(full == 0, 2, 3)))
        base = sl.rl.cell_ceiling(s["core"], cn, pnr, mp)[0] >= mc
        ppk = np.array([pnr[m].max() if m.any() else 0.0 for m in s["core"]])
        marg.setdefault(s["group"], []).append((ppk - mp, full))
        row = dict(name=s["name"], group=s["group"], mc=mc, mp=mp, n=s["n"], base=base.copy(), full=full)
        for dp in DPS:
            ok = sl.rl.cell_ceiling(s["core"], cn, pnr, mp + dp)[0] >= mc
            lost = base & ~ok
            r = res.setdefault(s["group"], {}).setdefault(dp, {c: [0, 0] for c in CATS})
            for ci, c in enumerate(CATS):
                r[c][0] += int((lost & (cat == ci)).sum())
                r[c][1] += int((base & (cat == ci)).sum())
            kl = np.where(lost & (full == 1))[0]
            covered = sum(1 for i in kl if ((s["J"][i] >= 0.5) & ok).any())
            cov.setdefault(s["group"], {}).setdefault(dp, [0])[0] += covered
            row[dp] = (int((lost & (full == 1)).sum()), int((lost & (full == 0)).sum()), int((lost & (full == -1)).sum()), covered)
        per.append(row)

    print("\n== per session: candidates seedable from their own core at the run's thresholds, and what min_pnr + 2 removes ==")
    print("session | group | run min_corr / min_pnr | N | seedable kept / reviewed deletes / auto-rejected | "
          "+2 loses: kept (covered by a surviving copy) / reviewed deletes / auto-rejected")
    for r in per:
        b, f = r["base"], r["full"]
        print("  %s | %s | %.2f / %.1f | %d | %d of %d / %d of %d / %d of %d | %d (%d) / %d / %d" % (
            r["name"][:24], r["group"], r["mc"], r["mp"], r["n"], int((b & (f == 1)).sum()), int((f == 1).sum()),
            int((b & (f == 0)).sum()), int((f == 0).sum()), int((b & (f == -1)).sum()), int((f == -1).sum()),
            r[2][0], r[2][3], r[2][1], r[2][2]))

    def block(title, groups):
        print("\n  %s" % title)
        print("    d | kept lost (of them covered by a surviving copy) | reviewed deletes lost: dup-of-kept, other | "
              "auto-rejected lost | every non-kept candidate lost (the 11-11 'rejected') | all candidates lost")
        for dp in DPS:
            t = {c: [0, 0] for c in CATS}
            cv = 0
            for g in groups:
                if g not in res:
                    continue
                for c in CATS:
                    t[c][0] += res[g][dp][c][0]
                    t[c][1] += res[g][dp][c][1]
                cv += cov[g][dp][0]
            nk = [sum(t[c][0] for c in CATS[1:]), sum(t[c][1] for c in CATS[1:])]
            al = [sum(t[c][0] for c in CATS), sum(t[c][1] for c in CATS)]
            print("   +%d | %3d of %4d (%s), covered %d | %3d of %3d (%s), %3d of %4d (%s) | %3d of %4d (%s) | "
                  "%4d of %4d (%s) | %4d of %4d (%s)" % (
                      dp, t["kept"][0], t["kept"][1], pct(*t["kept"]), cv, t["del_dup"][0], t["del_dup"][1], pct(*t["del_dup"]),
                      t["del_other"][0], t["del_other"][1], pct(*t["del_other"]), t["auto"][0], t["auto"][1], pct(*t["auto"]),
                      nk[0], nk[1], pct(*nk), al[0], al[1], pct(*al)))

    print("\n== raising min_pnr above the run's value, by group ==")
    for g in sl.GROUP_ORDER:
        if g in res:
            block(g, [g])
    block("ALL sessions run at 0.70 (11 pilots + Taylor's 12): the bright set", ["pilots at 0.70 (11)", "Taylor 09xx at 0.70 (12)"])
    block("the 12 pilots as on 11-11 (11 at 0.70 + 082726-pnb105 at 0.60): must reproduce +2 = 3 of 410 kept, 253 of 1884 rejected",
          ["pilots at 0.70 (11)", "pilot 0.60 (082726-pnb105)"])
    block("ALL six Yawen pnb105 sessions at 0.60", ["Yawen pnb105 FOVA at 0.60 (4)", "Yawen pnb105 FOVC 090526 at 0.60",
                                                   "Yawen pnb105 FOVB 090626 at 0.60"])
    block("ALL 31 labelled sessions", sl.GROUP_ORDER)

    print("\n== kept-rate by PNR margin (max PNR in the footprint core minus the run's min_pnr) ==")
    print("   columns per margin band: kept of reviewed (auto-rejected candidates in the band, never shown)")
    bands = ((-99, 0), (0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 999))
    for title, groups in (("pilots at 0.70 (11)", ["pilots at 0.70 (11)"]), ("Taylor 09xx at 0.70 (12)", ["Taylor 09xx at 0.70 (12)"]),
                          ("Yawen pnb105 FOVA at 0.60 (4)", ["Yawen pnb105 FOVA at 0.60 (4)"]),
                          ("the other four sessions at 0.60", ["pilot 0.60 (082726-pnb105)", "Taylor 090526-pnb99 dim FOVB at 0.60",
                                                               "Yawen pnb105 FOVC 090526 at 0.60", "Yawen pnb105 FOVB 090626 at 0.60"])):
        m = np.concatenate([a for g in groups for a, _ in marg.get(g, [])])
        f = np.concatenate([b for g in groups for _, b in marg.get(g, [])])
        cells = []
        for lo, hi in bands:
            sel = (m >= lo) & (m < hi)
            rev = int((sel & (f >= 0)).sum())
            kk = int((sel & (f == 1)).sum())
            cells.append("%s to %s: %d of %d (%s; auto %d)" % (lo if lo > -99 else "<", hi if hi < 999 else ">", kk, rev,
                                                              pct(kk, rev), int((sel & (f == -1)).sum())))
        print("  %-32s %s" % (title, " | ".join(cells)))


if __name__ == "__main__":
    main()
