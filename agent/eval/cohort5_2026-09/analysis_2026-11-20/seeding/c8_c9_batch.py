"""C8 / C9 of the 2026-11-11 red team on every labelled cohort-5 6odor session (2026-11-20:
12 threshold-0 pilots, Taylor's 13 and Yawen's 6 of the 28-session batch).  Read-only.

Definitions carried over from redteam/c8_c9_pilots.py:
  ceiling   highest min_corr at which a candidate can still seed from inside its own core
            (core = footprint >= half its max at 256 x 256), given the run's min_pnr, on the
            greedyROI seeding image of the full movie (11-11 cache).  An ESTIMATE of what a
            higher min_corr would have found: it is built on the first seeding image only.
  rule      floor + 0.444 * (pk50 - floor), the per-session proposal under test (C8).
Added here:
  labels    1 kept, 0 reviewed delete, -1 auto-rejected at 0.01 (never shown; own category)
  dup       reviewed delete, not motion-tagged, whose support (> 0.2 max, 512 x 512) has
            IoU >= 0.5 with a candidate KEPT in the same session (a second copy of a kept cell)
  sections  1 per-session C8 table   2 kept-rate by ceiling band per group
            3 C9: the sessions seeded at 0.60, low-ceiling candidates and the kept cells only
              the low seed could find   4 the three sessions with an archived 0.70 run:
              the ceiling estimate against what 0.70 really produced
Usage: c8_c9_batch.py > c8_c9_batch.txt
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sd_lib as sl  # noqa: E402

BANDS =((0, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 0.85),
         (0.85, 0.90), (0.90, 1.01))


def pct(a, b):
    return "%.1f%%" % (100.0 * a / b) if b else "-"


def main():
    S = sl.sessions()
    print("labelled sessions: %d | by reviewer: %s" % (len(S), ", ".join(
        "%s %d" % (r, sum(1 for s in S if s["reviewer"] == r)) for r in sorted({s["reviewer"] for s in S}))))
    print("candidates %d, kept %d, reviewed deletes %d (dup-of-kept %d), auto-rejected (never shown) %d" % (
        sum(s["n"] for s in S), sum(int((s["full"] == 1).sum()) for s in S), sum(int((s["full"] == 0).sum()) for s in S),
        sum(int(s["dup"].sum()) for s in S), sum(int((s["full"] == -1).sum()) for s in S)))
    y6 = [s for s in S if s["reviewer"] == "Yawen"]
    print("check against reviewer_check_yawen.txt section G: Yawen's six sessions, dup-of-kept deletes %d of %d deletes" % (
        sum(int(s["dup"].sum()) for s in y6), sum(int((s["full"] == 0).sum()) for s in y6)))
    print("label join (final footprints whose best-cosine candidate is a KEEP): " + "; ".join(
        "%s %d/%d" % (s["name"][:13], s["join_keep"], s["n_final"]) for s in S if s["join_keep"] != s["n_final"]) +
        " | all others exact")

    # ---------------------------------------------------------------- 1
    print("\n== 1. C8 per session ==")
    print("session | group | reviewer | ran at / min_pnr | T | floor | pk50 | rule (f 0.444) | N / auto / reviewed / kept | "
          "kept with ceiling < rule | kept with ceiling < the run's min_corr | kept with ceiling < 0.70 | "
          "kept ceiling p5 / p10 / p25 | reviewed with ceiling < 0.70 | dup deletes")
    tot = {}
    for s in S:
        k = s["full"] == 1
        rev = s["full"] >= 0
        ce = s["ce"]
        t = tot.setdefault(s["group"], dict(n=0, sess=0, kept=0, below_rule=0, below_run=0, below70=0, rev=0, rev70=0,
                                            auto=0, dup=0))
        t["sess"] += 1
        t["n"] += s["n"]
        t["kept"] += int(k.sum())
        t["below_rule"] += int((ce[k] < s["rule"]).sum())
        t["below_run"] += int((ce[k] < s["mc"]).sum())
        t["below70"] += int((ce[k] < 0.70).sum())
        t["rev"] += int(rev.sum())
        t["rev70"] += int((rev & (ce < 0.70)).sum())
        t["auto"] += int((s["full"] == -1).sum())
        t["dup"] += int(s["dup"].sum())
        print("  %s | %s | %s | %.2f / %.1f | %d | %.3f | %.3f | %.2f | %d / %d / %d / %d | %d | %d | %d | %.3f / %.3f / %.3f | %d | %d" % (
            s["name"][:24], s["group"], s["reviewer"], s["mc"], s["mp"], s["T"], s["floor"], s["pk50"], s["rule"],
            s["n"], int((s["full"] == -1).sum()), int(rev.sum()), int(k.sum()), int((ce[k] < s["rule"]).sum()),
            int((ce[k] < s["mc"]).sum()), int((ce[k] < 0.70).sum()), *np.percentile(ce[k], [5, 10, 25]),
            int((rev & (ce < 0.70)).sum()), int(s["dup"].sum())))
    print("\n  pooled by group:")
    for g in sl.GROUP_ORDER:
        if g not in tot:
            continue
        t = tot[g]
        print("   %-38s sessions %2d  candidates %4d  auto %4d  reviewed %4d  kept %3d | kept with ceiling < rule %d (%s) | "
              "< run's min_corr %d (%s) | < 0.70 %d (%s) | reviewed with ceiling < 0.70: %d (%s of reviewed) | dup deletes %d" % (
                  g, t["sess"], t["n"], t["auto"], t["rev"], t["kept"], t["below_rule"], pct(t["below_rule"], t["kept"]),
                  t["below_run"], pct(t["below_run"], t["kept"]), t["below70"], pct(t["below70"], t["kept"]),
                  t["rev70"], pct(t["rev70"], t["rev"]), t["dup"]))
    allk = sum(t["kept"] for t in tot.values())
    print("   all 31: kept %d | ceiling < rule %d (%s) | ceiling < the run's min_corr %d (%s)" % (
        allk, sum(t["below_rule"] for t in tot.values()), pct(sum(t["below_rule"] for t in tot.values()), allk),
        sum(t["below_run"] for t in tot.values()), pct(sum(t["below_run"] for t in tot.values()), allk)))
    p70 = [s for s in S if s["mc"] >= 0.65]
    k70 = sum(int((s["full"] == 1).sum()) for s in p70)
    lost = sum(int(((s["full"] == 1) & (s["ce"] < s["rule"])).sum()) for s in p70)
    print("   sessions run at 0.70 whose rule value is ABOVE 0.70 (the rule would raise them): %s" % ", ".join(
        "%s %.2f (kept with ceiling < rule: %d of %d)" % (s["name"][:13], s["rule"], int(((s["full"] == 1) & (s["ce"] < s["rule"])).sum()),
                                                         int((s["full"] == 1).sum())) for s in p70 if s["rule"] > 0.70))
    print("   sessions run at 0.70, all %d: kept %d, kept with ceiling < rule %d (%s)" % (len(p70), k70, lost, pct(lost, k70)))

    # ---------------------------------------------------------------- 2
    print("\n== 2. kept-rate by seeding-ceiling band, per group ==")
    print("   columns: candidates | auto-rejected (never shown) | reviewed | kept | dup-of-kept deletes | other deletes |")
    print("            kept / reviewed | kept / (reviewed - dup)")
    for g in sl.GROUP_ORDER:
        ss = [s for s in S if s["group"] == g]
        if not ss:
            continue
        ce = np.concatenate([s["ce"] for s in ss])
        full = np.concatenate([s["full"] for s in ss])
        dup = np.concatenate([s["dup"] for s in ss])
        print("  %s: sessions %d, candidates %d, kept %d" % (g, len(ss), len(full), int((full == 1).sum())))
        for lo, hi in BANDS:
            m = (ce >= lo) & (ce < hi)
            if not m.any():
                continue
            rev = int((m & (full >= 0)).sum())
            kk = int((m & (full == 1)).sum())
            dd = int((m & dup).sum())
            print("    %.2f-%.2f: %4d | %4d | %4d | %3d | %3d | %4d | %6s | %6s" % (
                lo, min(hi, 1.0), int(m.sum()), int((m & (full == -1)).sum()), rev, kk, dd, rev - kk - dd,
                pct(kk, rev), pct(kk, rev - dd)))

    # ---------------------------------------------------------------- 3
    print("\n== 3. C9: sessions seeded at 0.60 ==")
    print("   low-ceiling candidates and what became of them.  'copy of a >= 0.70 kept cell' = not kept, IoU >= 0.5 with a")
    print("   KEPT candidate whose own ceiling is >= 0.70 (the low seed added a second copy, not a new cell).")
    print("   'only-0.60 cell' = KEPT, ceiling < 0.70, and no other candidate with ceiling >= 0.70 overlaps it at IoU >= 0.5")
    print("   (in brackets: the same count with the looser IoU >= 0.2).")
    agg = {}
    for s in [s for s in S if s["mc"] < 0.65]:
        ce, full, J = s["ce"], s["full"], s["J"]
        hi_any = ce >= 0.70
        hi_kept = hi_any & (full == 1)
        ov_hi_kept = J[:, hi_kept].max(1) if hi_kept.any() else np.zeros(s["n"])
        ov_hi_any = J[:, hi_any].max(1) if hi_any.any() else np.zeros(s["n"])
        print("  %s | %s | %s | pk50 %.3f | rule %.2f | N %d, auto %d, reviewed %d, kept %d" % (
            s["name"][:24], s["group"], s["reviewer"], s["pk50"], s["rule"], s["n"], int((full == -1).sum()),
            int((full >= 0).sum()), int((full == 1).sum())))
        for nm, m in (("ceiling < 0.60", ce < 0.60), ("0.60 <= ceiling < 0.70", (ce >= 0.60) & (ce < 0.70)),
                      ("0.60 <= ceiling < rule %.2f" % s["rule"], (ce >= 0.60) & (ce < s["rule"])),
                      ("ceiling >= 0.70", ce >= 0.70)):
            notk = m & (full != 1)
            cp = notk & (ov_hi_kept >= 0.5)
            print("      %-28s candidates %3d: auto-rejected %3d, reviewed %3d, KEPT %2d | of the not kept: copy of a >= 0.70 kept "
                  "cell %3d (reviewed %d, auto %d), other %3d" % (
                      nm, int(m.sum()), int((m & (full == -1)).sum()), int((m & (full >= 0)).sum()), int((m & (full == 1)).sum()),
                      int(cp.sum()), int((cp & (full == 0)).sum()), int((cp & (full == -1)).sum()), int((notk & ~cp).sum())))
        lowk = np.where((full == 1) & (ce < 0.70))[0]
        only = [i for i in lowk if ov_hi_any[i] < 0.5]
        only2 = [i for i in lowk if ov_hi_any[i] < 0.2]
        print("      kept with ceiling < 0.70: %d of %d kept; only-0.60 cells: %d (%d); their ceilings: %s" % (
            len(lowk), int((full == 1).sum()), len(only), len(only2), np.round(np.sort(ce[only]), 3).tolist()))
        a = agg.setdefault(s["group"], dict(sess=0, kept=0, lowk=0, only=0, only2=0, low=0, lowrev=0, dup=0, dels=0, n=0, auto=0))
        a["sess"] += 1
        a["kept"] += int((full == 1).sum())
        a["lowk"] += len(lowk)
        a["only"] += len(only)
        a["only2"] += len(only2)
        a["low"] += int((ce < 0.70).sum())
        a["lowrev"] += int(((ce < 0.70) & (full >= 0)).sum())
        a["dup"] += int(s["dup"].sum())
        a["dels"] += int((full == 0).sum())
        a["n"] += s["n"]
        a["auto"] += int((full == -1).sum())
    print("\n  by group (0.60 sessions):")
    for g in sl.GROUP_ORDER:
        if g not in agg:
            continue
        a = agg[g]
        print("   %-38s sessions %d  candidates %4d  kept %3d | candidates with ceiling < 0.70: %3d (reviewed %3d) | kept among them "
              "%2d (%s of kept) | only-0.60 cells %2d (%d at IoU 0.2) | dup deletes %3d of %4d deletes (%.2f per kept)" % (
                  g, a["sess"], a["n"], a["kept"], a["low"], a["lowrev"], a["lowk"], pct(a["lowk"], a["kept"]), a["only"],
                  a["only2"], a["dup"], a["dels"], a["dup"] / max(a["kept"], 1)))
    print("  for scale, sessions run at 0.70 (dup deletes per kept cell):")
    for g in ("pilots at 0.70 (11)", "Taylor 09xx at 0.70 (12)"):
        ss = [s for s in S if s["group"] == g]
        d = sum(int(s["dup"].sum()) for s in ss)
        k = sum(int((s["full"] == 1).sum()) for s in ss)
        dl = sum(int((s["full"] == 0).sum()) for s in ss)
        print("   %-38s dup deletes %3d of %4d deletes, kept %3d (%.2f per kept)" % (g, d, dl, k, d / max(k, 1)))
    for s in S:
        if s["animal"] == "pnb105":
            print("   pnb105 %s (ran at %.2f, %s): dup deletes %3d of %3d deletes, kept %2d (%.2f per kept)" % (
                s["name"][:13], s["mc"], s["reviewer"], int(s["dup"].sum()), int((s["full"] == 0).sum()),
                int((s["full"] == 1).sum()), s["dup"].sum() / max(int((s["full"] == 1).sum()), 1)))

    # ---------------------------------------------------------------- 4
    print("\n== 4. measured, not estimated: the three sessions that also have an archived 0.70 run ==")
    print("   a live (0.60) candidate is 'present at 0.70' when its support has IoU >= 0.5 with a candidate of the archived")
    print("   0.70 run (<session>/.prev_mincorr070_*/spatial_footprints.mat; same movie, same code, min_corr 0.70).")
    for s in S:
        prev = sorted(p for p in s["sd"].iterdir() if p.is_dir() and p.name.startswith(".prev_mincorr070"))
        if not prev or s["mc"] >= 0.65:
            continue
        old = sl.rl.load_sf(prev[-1] / "spatial_footprints.mat")
        Mo = sl.csr(sl.support_coords(old), len(old))
        Mn = sl.csr(s["co"], s["n"])
        Jo = sl.iou(Mn, Mo)
        present = Jo.max(1) >= 0.5
        full, ce = s["full"], s["ce"]
        print("  %s (%s, %s): live 0.60 set %d candidates, archived 0.70 set %d (%s); 0.70 candidates matched by a live one: %d" % (
            s["name"][:24], s["group"], s["reviewer"], s["n"], len(old), prev[-1].name, int((Jo.max(0) >= 0.5).sum())))
        for nm, m in (("present at 0.70", present), ("ADDED by 0.60", ~present)):
            print("      %-16s %3d candidates: auto-rejected %3d, reviewed %3d, KEPT %2d, dup-of-kept deletes %3d, other deletes %3d | "
                  "ceiling >= 0.70 in %3d, < 0.70 in %3d" % (
                      nm, int(m.sum()), int((m & (full == -1)).sum()), int((m & (full >= 0)).sum()), int((m & (full == 1)).sum()),
                      int((m & s["dup"]).sum()), int((m & (full == 0) & ~s["dup"]).sum()), int((m & (ce >= 0.70)).sum()),
                      int((m & (ce < 0.70)).sum())))
        k = full == 1
        print("      kept cells: present at 0.70 and ceiling >= 0.70: %d | present, ceiling < 0.70: %d | added, ceiling >= 0.70: %d | "
              "added, ceiling < 0.70: %d   (ceiling as a predictor of 'a 0.70 run finds it': agrees on %d of %d kept cells)" % (
                  int((k & present & (ce >= 0.70)).sum()), int((k & present & (ce < 0.70)).sum()),
                  int((k & ~present & (ce >= 0.70)).sum()), int((k & ~present & (ce < 0.70)).sum()),
                  int((k & (present == (ce >= 0.70))).sum()), int(k.sum())))
        ad = np.where(k & ~present)[0]
        print("      kept cells added by 0.60: ceilings %s" % np.round(np.sort(ce[ad]), 3).tolist())
        allc = ~present
        print("      all candidates added by 0.60: ceiling >= 0.70 in %d of %d (the ceiling estimate would have called these "
              "'found at 0.70')" % (int((allc & (ce >= 0.70)).sum()), int(allc.sum())))
        # one-to-one: a 0.70 candidate can stand for only ONE live candidate, so a second live copy of the
        # same object shows up as 'extra copy' instead of hiding under 'present'
        cost = np.where(Jo >= 0.5, -Jo, 0.0)
        ri, ci = linear_sum_assignment(cost)
        one = np.zeros(s["n"], bool)
        one[ri[Jo[ri, ci] >= 0.5]] = True
        extra = ~one & present
        newloc = ~present
        Jold = sl.iou(Mo, Mo)
        np.fill_diagonal(Jold, 0.0)
        print("      one-to-one matching (each 0.70 candidate stands for one live candidate); the archived 0.70 set itself holds "
              "%d candidates that overlap another 0.70 candidate at IoU >= 0.5:" % int((Jold.max(1) >= 0.5).sum()))
        for nm, m in (("matched 1:1", one), ("extra copy at 0.60", extra), ("new location at 0.60", newloc)):
            print("        %-22s %3d candidates: auto-rejected %3d, reviewed %3d, KEPT %2d, dup-of-kept deletes %3d, other deletes %3d" % (
                nm, int(m.sum()), int((m & (full == -1)).sum()), int((m & (full >= 0)).sum()), int((m & (full == 1)).sum()),
                int((m & s["dup"]).sum()), int((m & (full == 0) & ~s["dup"]).sum())))


if __name__ == "__main__":
    main()
