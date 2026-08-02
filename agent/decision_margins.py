"""
decision_margins.py

Answers a specific question: can a ~1e-12 perturbation of the background estimate
change which cells come out of a headless run?

A perturbation that small cannot drift the result — every quantity stays equal to
~13 significant figures.  It can only matter if it flips a DISCRETE comparison.
So the useful measurement is not "how big is the perturbation" but "how close did
this session's actual decisions come to their thresholds".

The comparisons that decide the candidate set:

  quickMerge.m:54   (A_overlap > 0.1) & (C_corr > 0.85) & (S_corr > 0)
                    -- note A_overlap is built from the BINARY mask A>0, not from
                    A's values, so it only moves if a footprint pixel crosses zero
  trimSpatial       zeroes footprint values below a fraction of each neuron's max
  initComponents /  seeds where the correlation and PNR images exceed min_corr and
  pickNeurons       min_pnr

For each, we report the smallest margin observed and how many orders of magnitude
that sits above the perturbation.  Reads only exports already on disk.

    python decision_margins.py <session_dir> [--min-corr 0.87] [--min-pnr 12.4]
                               [--perturbation 5.14e-12]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

# Thresholds as the pipeline sets them (CNMFe_Biane_headless.m:53, :134)
MERGE_A_THR = 1e-1
MERGE_C_THR = 0.85
MERGE_S_THR = 0.0
TRIM_FRAC = 0.01          # neuron.trimSpatial(0.01, 3)


def _corr_rows(M: np.ndarray) -> np.ndarray:
    """Row-wise correlation matrix, matching MATLAB corr(M')."""
    M = M - M.mean(axis=1, keepdims=True)
    sd = np.sqrt((M ** 2).sum(axis=1))
    sd[sd == 0] = np.inf                      # constant rows -> zero correlation
    Mn = M / sd[:, None]
    return Mn @ Mn.T


def _margin_report(name: str, margins: np.ndarray, thr_desc: str,
                   perturbation: float) -> str:
    if margins.size == 0:
        return f"  {name:<34} no comparisons of this kind"
    m = float(np.min(margins))
    if m == 0.0:
        return (f"  {name:<34} min margin 0 (exact tie!) vs {thr_desc} "
                f"— a tie CAN flip")
    orders = np.log10(m / perturbation)
    verdict = "safe" if orders >= 3 else "*** TOO CLOSE ***"
    return (f"  {name:<34} min margin {m:.3e} vs {thr_desc}  "
            f"= {orders:5.1f} orders above perturbation  [{verdict}]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--min-corr", type=float, default=0.87)
    ap.add_argument("--min-pnr", type=float, default=12.4)
    ap.add_argument("--perturbation", type=float, default=5.14e-12,
                    help="measured max|diff| of the candidate change")
    args = ap.parse_args()

    sd = args.session_dir
    pert = args.perturbation

    print(f"Decision-margin analysis — {sd.name}")
    print(f"Perturbation under test: {pert:.3e} (absolute, in Yest units)\n")

    # ---- footprints ----
    sf = sio.loadmat(str(sd / "spatial_footprints.mat"))["spatial_footprints"]
    N = sf.shape[0]
    A = sf.reshape(N, -1).T                    # (pixels, N), matching neuron.A
    print(f"{N} candidates, {A.shape[0]} pixels")

    C = np.loadtxt(sd / "C.txt")
    S = np.loadtxt(sd / "S.txt")
    if C.ndim == 1:
        C = C[None, :]
    if S.ndim == 1:
        S = S[None, :]
    print(f"C {C.shape}, S {S.shape}\n")

    lines = []

    # ---- 1. trimSpatial: how close is any surviving footprint value to the
    #         per-neuron cutoff?  This is the only route by which A>0 can change.
    trim_margins = []
    for i in range(N):
        col = A[:, i]
        nz = col[col > 0]
        if nz.size == 0:
            continue
        cutoff = TRIM_FRAC * col.max()
        trim_margins.append(np.abs(nz - cutoff).min())
    lines.append(_margin_report("trimSpatial value vs cutoff",
                                np.array(trim_margins),
                                f"{TRIM_FRAC:g}*max per neuron", pert))

    # Distance of nonzero footprint values from zero — flipping A>0 the other way.
    nzmin = A[A > 0].min() if (A > 0).any() else 0.0
    lines.append(_margin_report("smallest nonzero footprint value",
                                np.array([nzmin]), "0 (the A>0 mask)", pert))

    # ---- 2. merge criterion ----
    mask = (A > 0).astype(float)
    cnt = mask.sum(axis=0)
    cnt[cnt == 0] = np.inf
    tempn = mask / np.sqrt(cnt)
    A_overlap = tempn.T @ tempn
    np.fill_diagonal(A_overlap, 0.0)

    C_corr = _corr_rows(C)
    np.fill_diagonal(C_corr, 0.0)
    S_corr = _corr_rows(S)
    np.fill_diagonal(S_corr, 0.0)

    iu = np.triu_indices(N, k=1)
    ao, cc, sc = A_overlap[iu], C_corr[iu], S_corr[iu]

    lines.append(_margin_report("A_overlap vs merge threshold",
                                np.abs(ao - MERGE_A_THR),
                                f"{MERGE_A_THR:g}", pert))

    # C_corr only decides for pairs that already clear the spatial gate.
    gate_a = ao > MERGE_A_THR
    lines.append(_margin_report("C_corr vs 0.85 (spatial gate passed)",
                                np.abs(cc[gate_a] - MERGE_C_THR),
                                f"{MERGE_C_THR:g}", pert))

    # S_corr>0 is the threshold nearest to the bulk of the distribution, but it
    # only decides for pairs that already cleared BOTH other gates.
    gate_ac = gate_a & (cc > MERGE_C_THR)
    lines.append(_margin_report("S_corr vs 0 (both gates passed)",
                                np.abs(sc[gate_ac] - MERGE_S_THR),
                                f"{MERGE_S_THR:g}", pert))
    print(f"pairs clearing spatial gate: {int(gate_a.sum())} of {ao.size}")
    print(f"pairs clearing spatial+temporal gates: {int(gate_ac.sum())}\n")

    # ---- 3. seeding thresholds ----
    Cn = sio.loadmat(str(sd / "Cn.mat"))["Cn"].ravel()
    pnr = sio.loadmat(str(sd / "pnr.mat"))["pnr"].ravel()
    Cn = Cn[np.isfinite(Cn)]
    pnr = pnr[np.isfinite(pnr)]
    lines.append(_margin_report("Cn vs min_corr", np.abs(Cn - args.min_corr),
                                f"{args.min_corr:g}", pert))
    lines.append(_margin_report("pnr vs min_pnr", np.abs(pnr - args.min_pnr),
                                f"{args.min_pnr:g}", pert))

    print("Decision margins (smallest observed):")
    for ln in lines:
        print(ln)

    print("\nCaveat: these are the margins at the FINAL state.  A run performs the")
    print("merge test ~8 times on intermediate states that are not saved, so this")
    print("is representative rather than exhaustive.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
