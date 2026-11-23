"""Driver: run the read-only missed-cell diagnostic on every red-guide session.
Reads D:\\Julian_CNMFe and D:\\retro_tdTomato_Alignment; writes only under the scratchpad miss/ folder.
Deletes only its own cache file (miss/<tag>/ds.npy) after each session.
"""
import json, re, subprocess, sys, time
from pathlib import Path

PY = sys.executable
HERE = Path(__file__).parent
OUT = HERE / "miss"
RES = Path(r"D:\retro_tdTomato_Alignment\results")

guides = sorted(RES.glob("*/*/*/red_guide/guide.json"))
print(f"{len(guides)} red-guide sessions", flush=True)
for gj in guides:
    t0 = time.time()
    g = json.load(open(gj))
    sess = g["session"]; S = Path(g["acorn_dir"]); G = gj.parent
    animal, date = gj.parts[-5], gj.parts[-4]
    tag = f"{date}-{animal}"; C = OUT / tag; C.mkdir(parents=True, exist_ok=True)
    if (C / "reds.json").exists() and (C / "blobs.json").exists() and (C / "done.txt").exists():
        print(f"[skip] {tag} already done", flush=True); continue
    log = (S / "agent_run.log").read_text(encoding="utf-8", errors="replace")
    fp = re.findall(r"Final parameters: \{'gSig': (\d+), 'gSiz': (\d+), 'min_corr': ([\d.]+), 'min_pnr': ([\d.]+)", log)
    if not fp:
        print(f"[FAIL] {tag}: no Final parameters line", flush=True); continue
    gsig, gsiz, mc, mp = fp[-1]
    with open(C / "run.txt", "w") as f:
        f.write(f"{sess}\nmin_corr {mc} min_pnr {mp} gSig {gsig} gSiz {gsiz}\n")
        if not (C / "ds.npy").exists():
            r = subprocess.run([PY, "-I", str(HERE / "cache_movie.py"), str(S / f"{sess}.mat"), str(C)],
                               capture_output=True, text=True)
            f.write("CACHE " + r.stdout[-300:] + r.stderr[-2000:] + "\n")
            if r.returncode != 0:
                print(f"[FAIL] {tag}: cache rc {r.returncode}: {r.stderr[-400:]}", flush=True); continue
        ok = True
        for script in ["missed_cells.py", "red_somata.py"]:
            r = subprocess.run([PY, "-I", str(HERE / script), str(S), str(G), str(C), mc, mp],
                               capture_output=True, text=True)
            f.write(f"=== {script} rc {r.returncode}\n{r.stdout}\n{r.stderr[-3000:]}\n")
            if r.returncode != 0:
                ok = False; print(f"[FAIL] {tag}: {script} rc {r.returncode}: {r.stderr[-400:]}", flush=True)
    if ok:
        (C / "done.txt").write_text("ok\n")
        (C / "ds.npy").unlink()          # own cache only
    summ = [l for l in (C / "run.txt").read_text().splitlines() if l.startswith("SUMMARY")]
    print(f"[{'ok' if ok else 'FAIL'}] {tag} min_corr {mc} ({time.time()-t0:.0f} s) {summ[0][8:] if summ else ''}", flush=True)
print("ALL DONE", flush=True)
