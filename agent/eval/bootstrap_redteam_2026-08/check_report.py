"""
check_report.py -- completeness critic for redteam_report.md.

  * 19 attack rows with one of PASS / FAIL / INCONCLUSIVE;
  * every attack has a results JSON and a refuter JSON;
  * every numeric token in the report that looks like a measurement appears
    in some results JSON (crude: numbers with a decimal point, 3+ significant
    digits; dates / line numbers / years whitelisted);
  * the deploy verdicts and the ranked recommendation list are present;
  * every results JSON carries the current pin hash;
  * no 'expect', 'should be', 'probably' stated as a measurement.
Exit 1 on any failure.
"""
import json
import re
import sys

import rt_lib as rt

REPORT = rt.SP / "redteam_report.md"
ATTACKS = [f"a{i:02d}" for i in range(1, 20)]


def all_numbers_in_results():
    nums = set()
    for p in rt.RESULTS.glob("*.json"):
        txt = p.read_text()
        for m in re.finditer(r"-?\d+\.\d+", txt):
            v = float(m.group())
            for prec in (1, 2, 3, 4):
                nums.add(round(v, prec))
                nums.add(round(v * 100, prec))     # percentages
                nums.add(round(v / 100, prec))
    return nums


def main():
    ok = True
    txt = REPORT.read_text(encoding="utf-8")
    pin = rt.pin_hash()
    # 1. verdict rows
    missing = []
    for a in ATTACKS:
        n = int(a[1:])
        if not re.search(rf"^\|\s*{n}\s*\|.*\*\*(PASS|FAIL|INCONCLUSIVE)\*\*", txt, re.M):
            missing.append(a)
    print(f"verdict rows: {19 - len(missing)}/19" + (f"  MISSING {missing}" if missing else ""))
    ok &= not missing
    # 2. json + refuter present, pinned
    for a in ATTACKS:
        r = rt.RESULTS / f"{a}.json"
        f = rt.RESULTS / f"{a}_refute.json"
        if not r.exists() or not f.exists():
            print(f"  {a}: results {r.exists()} refuter {f.exists()}")
            ok = False
            continue
        if json.loads(r.read_text()).get("pin_hash") != pin:
            print(f"  {a}: pin hash mismatch")
            ok = False
    # 3. numeric traceability
    nums = all_numbers_in_results()
    untraced = []
    for m in re.finditer(r"(?<![\w.])(-?\d+\.\d{2,4})(?![\w.])", txt):
        v = float(m.group(1))
        if abs(v) > 3000 or v in (2026.0,):
            continue
        if round(v, 4) not in nums and round(v, 3) not in nums and round(v, 2) not in nums:
            untraced.append(m.group(1))
    untraced = sorted(set(untraced))
    print(f"numeric tokens not traceable to results/*.json: {len(untraced)}" + (f" -> {untraced[:30]}" if untraced else ""))
    ok &= len(untraced) == 0
    # 4. sections
    for sec in ("Verdict on each deploy", "Ranked recommendations", "Section-D", "coverage", "Open items"):
        present = sec.lower() in txt.lower()
        print(f"section '{sec}': {'present' if present else 'MISSING'}")
        ok &= present
    # 5. hedged measurements
    hedges = [m.group() for m in re.finditer(r"(expect(ed)? to be|should be|probably) \d", txt)]
    print(f"hedged numbers: {hedges}")
    ok &= not hedges
    print("CHECK_REPORT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
