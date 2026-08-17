"""
sweep_gsig.py -- measure the cost of a too-large gSig on a single session.

gSig cannot be read off the video (tested: against 31 animals with expert-set
gSig, apparent cell size gives r2 = 0.17 and mis-predicts by 1.3 units on
average).  But its consequence IS measurable: run the same session at several
gSig values and count the cells that a larger gSig fails to find.

That asymmetry is what matters here.  greedyROI_endoscope builds a
mean-subtracted Gaussian PSF (band-pass, surround out to gSiz) and detects on
the filtered data, so an oversized gSig suppresses small, densely packed cells.
A false positive lands in the review package and a human deletes it; a false
negative never enters neuron.mat and is invisible to every QC surface in the
pipeline -- viewNeurons offers keep/delete/split/trim but has no "add".

Each arm runs into DG_AL/.sweep/gSig{NN}/ .  The dot prefix keeps the whole
tree invisible to the watcher, the curator, REVIEW_QUEUE.md and every other
scanner, so a sweep can never be mistaken for real sessions.

Usage:
    python sweep_gsig.py                       # default arms, default session
    python sweep_gsig.py --gsig 4 5 6 7
    python sweep_gsig.py --session <path-to-session-dir>
    python sweep_gsig.py --compare-only        # skip CNMFe, just re-compare

Reference arm: the session's own completed run is used as-is (not re-run).
"""
import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))

import config_DG_AL
sys.modules["config"] = config_DG_AL

import params as param_estimator
import run_cnmfe
import features as feat_module

# Same constant the validated bootstrap matcher uses (bootstrap_preagent.py).
SPATIAL_MATCH_THRESHOLD = 0.45

DEFAULT_SESSION = (Path("d:/Julian_CNMFe/DG_AL/odor_encoding") /
                   "AVG4x-TSeries-040623-DG6D-356um-406um-2z-000A")

LOG_PATH = AGENT_DIR / "logs" / "sweep_gsig.log"


def log(msg=""):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def wait_for_watcher_idle(max_wait_min=90):
    """
    Hold off until the DG_AL watcher has no session mid-flight, so the sweep
    never becomes a fourth concurrent MATLAB job.  A session is mid-flight if
    its folder has a .tif but no neuron.mat yet.
    """
    root = config_DG_AL.DATA_ROOT
    deadline = time.time() + max_wait_min * 60
    while time.time() < deadline:
        busy = []
        for task in root.iterdir():
            if not task.is_dir() or task.name.startswith("."):
                continue
            for sd in task.iterdir():
                if not sd.is_dir():
                    continue
                has_tif = any(sd.glob("*.tif"))
                if has_tif and not (sd / "neuron.mat").exists():
                    busy.append(sd.name)
        if not busy:
            log("[SWEEP] Watcher idle -- starting.")
            return True
        log(f"[SWEEP] Waiting for watcher: {len(busy)} session(s) mid-flight "
            f"({', '.join(b[-28:] for b in busy)}). Re-checking in 120 s.")
        time.sleep(120)
    log(f"[SWEEP] WARNING: watcher still busy after {max_wait_min} min -- "
        f"proceeding anyway (expect contention for cores).")
    return False


def load_A(session_dir: Path):
    """Spatial footprints as (pixels x N), L2-normalised per column."""
    fps = feat_module.load_spatial(session_dir)          # (N, H, W)
    if fps is None or len(fps) == 0:
        return None, 0
    N = len(fps)
    A = fps.reshape(N, -1).T.astype(np.float64)          # (pixels, N)
    return A, N


def run_arm(session_dir: Path, sweep_root: Path, gSig: int, gSiz: int):
    """Pre-pass + param estimate + full CNMFe for one gSig, into its own dir."""
    session_name = session_dir.name
    arm_dir = sweep_root / f"gSig{gSig:02d}"
    arm_dir.mkdir(parents=True, exist_ok=True)

    if (arm_dir / "neuron.mat").exists():
        log(f"[SWEEP] gSig={gSig}: neuron.mat already present -- skipping run.")
        return arm_dir, None

    src_tif = session_dir / f"{session_name}.tif"
    dst_tif = arm_dir / f"{session_name}.tif"
    if not dst_tif.exists():
        if not src_tif.exists():
            log(f"[SWEEP] ERROR: source .tif missing: {src_tif}")
            return arm_dir, None
        log(f"[SWEEP] gSig={gSig}: copying .tif "
            f"({src_tif.stat().st_size/1e9:.2f} GB) into {arm_dir.name}/ ...")
        shutil.copy2(str(src_tif), str(dst_tif))

    log(f"[SWEEP] === arm gSig={gSig}, gSiz={gSiz} ===")
    t0 = time.time()
    if not run_cnmfe.run_pre_pass(session_name, arm_dir, gSig, gSiz, log):
        log(f"[SWEEP] gSig={gSig}: PRE-PASS FAILED.")
        return arm_dir, None

    p = param_estimator.estimate_all_params(session_name, arm_dir, log)
    # estimate_all_params re-derives gSig/gSiz from animal_params.json; this
    # sweep is precisely about overriding that.
    p["gSig"], p["gSiz"] = gSig, gSiz
    log(f"[SWEEP] gSig={gSig}: params {p}")

    if not run_cnmfe.run_full_cnmfe(session_name, arm_dir, p, log):
        log(f"[SWEEP] gSig={gSig}: FULL RUN FAILED.")
        return arm_dir, None
    log(f"[SWEEP] gSig={gSig}: done in {(time.time()-t0)/60:.1f} min.")
    return arm_dir, p


def compare(arms: dict, ref_label, out_pdf: Path):
    """
    For every arm, count the cells it finds that the reference arm does NOT,
    by max cosine similarity of spatial footprints (< threshold = unmatched).
    Renders the unmatched ones so they can be judged real or not.
    """
    loaded = {}
    for label, d in arms.items():
        A, N = load_A(d)
        if A is None:
            log(f"[COMPARE] {label}: no footprints found -- skipping.")
            continue
        loaded[label] = (A, N, d)
        log(f"[COMPARE] {label}: {N} candidates")

    if ref_label not in loaded:
        log(f"[COMPARE] reference arm {ref_label} unavailable -- cannot compare.")
        return
    A_ref, N_ref, _ = loaded[ref_label]
    nref = np.sqrt((A_ref ** 2).sum(axis=0)) + 1e-12

    log("")
    log(f"{'arm':<12}{'candidates':>12}{'matched in ref':>16}{'MISSED by ref':>16}")
    log("-" * 56)
    missed_sets = {}
    for label, (A, N, d) in loaded.items():
        if label == ref_label:
            log(f"{label:<12}{N:>12}{'-':>16}{'-':>16}")
            continue
        n = np.sqrt((A ** 2).sum(axis=0)) + 1e-12
        sim = (A / n).T @ (A_ref / nref)          # (N, N_ref)
        best = sim.max(axis=1)
        missed = np.nonzero(best < SPATIAL_MATCH_THRESHOLD)[0]
        missed_sets[label] = (missed, best, d)
        log(f"{label:<12}{N:>12}{N-len(missed):>16}{len(missed):>16}")

    log("")
    log(f"'MISSED by ref' = found at this gSig, no counterpart above cosine "
        f"{SPATIAL_MATCH_THRESHOLD} in {ref_label}.")
    log("These are the cells the larger gSig would silently lose. Judge them in the PDF.")

    _render(missed_sets, ref_label, out_pdf)


def _render(missed_sets, ref_label, out_pdf: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        log("[COMPARE] matplotlib unavailable -- skipping PDF.")
        return

    with PdfPages(str(out_pdf)) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        lines = ["gSig sweep -- cells missed by the reference arm", "",
                 f"Reference: {ref_label}", "",
                 f"Match rule: cosine similarity of spatial footprints,",
                 f"threshold {SPATIAL_MATCH_THRESHOLD} (same as bootstrap_preagent.py).", ""]
        for label, (missed, _, _) in missed_sets.items():
            lines.append(f"  {label}:  {len(missed)} cell(s) with no match in {ref_label}")
        lines += ["", "If these look like real neurons, the reference gSig is",
                  "losing them silently -- they never reach review."]
        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=12,
                va="top", family="monospace")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        for label, (missed, best, d) in missed_sets.items():
            if len(missed) == 0:
                continue
            fps = feat_module.load_spatial(d)
            try:
                traces = feat_module.load_traces(d)
            except Exception:
                traces = None
            for i in missed:
                fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
                fig.suptitle(f"{label} -- candidate {i+1}  "
                             f"(best match in {ref_label}: {best[i]:.2f})",
                             fontsize=12, fontweight="bold", color="crimson")
                axes[0].imshow(fps[i], cmap="hot", interpolation="nearest")
                axes[0].set_title("spatial footprint"); axes[0].axis("off")
                if traces is not None and i < len(traces):
                    axes[1].plot(traces[i], lw=0.6, color="black")
                    axes[1].set_xlabel("frame"); axes[1].set_ylabel("C_raw")
                else:
                    axes[1].axis("off")
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    log(f"[COMPARE] wrote {out_pdf}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    ap.add_argument("--gsig", type=int, nargs="+", default=[5, 7],
                    help="gSig arms to run (gSiz = 4x gSig, this lab's convention)")
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--no-wait", action="store_true",
                    help="don't wait for the watcher to go idle first")
    args = ap.parse_args()

    session_dir = args.session
    if not session_dir.is_dir():
        sys.exit(f"ERROR: session not found: {session_dir}")

    sweep_root = config_DG_AL.DATA_ROOT / ".sweep" / session_dir.name
    sweep_root.mkdir(parents=True, exist_ok=True)
    log("=" * 60)
    log(f"[SWEEP] session : {session_dir.name}")
    log(f"[SWEEP] arms    : gSig {args.gsig}  (+ existing run as reference)")
    log(f"[SWEEP] output  : {sweep_root}")
    log("=" * 60)

    if not args.compare_only:
        if not args.no_wait:
            wait_for_watcher_idle()
        for g in args.gsig:
            try:
                run_arm(session_dir, sweep_root, g, 4 * g)
            except Exception as e:
                import traceback
                log(f"[SWEEP] arm gSig={g} raised: {e}\n{traceback.format_exc()}")

    # Reference = the session's own completed run, at whatever gSig it used.
    arms = {}
    for g in args.gsig:
        d = sweep_root / f"gSig{g:02d}"
        if (d / "spatial_footprints.mat").exists():
            arms[f"gSig={g}"] = d
    ref_label = "gSig=9 (live)"
    arms[ref_label] = session_dir

    compare(arms, ref_label, sweep_root / "missed_by_reference.pdf")
    log("[SWEEP] complete.")


if __name__ == "__main__":
    main()
