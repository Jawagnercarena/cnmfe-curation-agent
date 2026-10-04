"""Temp-dir test for build_training_key.py (no real data, no MATLAB).

Builds synthetic sessions under a temp DATA_PARENT-like tree, a fake deployed
joblib, and checks: the review-column -> npz-row mapping for the three label
length cases, the missing motion_delete case, the arity-mismatch fallback,
the v2 hint rules, the 13-column refusal, the unsupported-area refusal, the
MATLAB round-trip classes of every key variable, and the spatial merge.
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import scipy.io as sio
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import training_common as tc
import build_training_key as bk

fails = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


def make_session(sd: Path, N: int, n_labels: int, auto, names, seed=0,
                 with_motion=True, video=False):
    rng = np.random.default_rng(seed)
    sd.mkdir(parents=True, exist_ok=True)
    F = len(names)
    X = rng.normal(size=(N, F))
    X[:, :13] = np.abs(X[:, :13]) + 0.1
    nb = 13
    if F == 35:
        from scipy.stats import rankdata
        X[:, nb:2 * nb] = rankdata(X[:, :nb], axis=0, method="average") / N
    np.savez(sd / "candidate_features.npz", feature_matrix=X,
             feature_names=np.array(names), auto_rejected=np.array(auto, dtype=int),
             n_candidates=np.array([N]))
    labels = (rng.random(n_labels) > 0.5).astype(np.uint8).reshape(-1, 1)
    d = {"labels": labels}
    if with_motion:
        motion = np.zeros_like(labels)
        if (labels == 0).any():
            motion[np.argmax(labels == 0)] = 1
        d["motion_delete"] = motion
    sio.savemat(sd / "labels.mat", d)
    (sd / "review_neuron.mat").write_bytes(b"not read by python")
    (sd / "labels_provenance.txt").write_text("reviewer: Taylor\ningested: x\nsource: y\n", encoding="utf-8")
    if video:
        (sd / f"{sd.name}.mat").write_bytes(b"video")
    return X


def make_model(model_dir: Path, n_features: int, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, n_features)); y = (X[:, 0] > 0).astype(int)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression().fit(sc.transform(X), y)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": sc, "clf": clf, "model_type": "logistic_regression",
                 "reject_threshold": 0.04, "n_features": n_features, "n_sessions": 7,
                 "feature_version": 2}, model_dir / "classifier.joblib")


tmp = Path(tempfile.mkdtemp(prefix="training_key_"))
try:
    names35 = bk.EXPECTED_NAMES
    check(len(names35) == 35 and names35[:13] == tc.BASE13, "EXPECTED_NAMES is the 35-column v2 contract")
    root = tmp / ".training"
    P = dict(bk.PARAM_DEFAULTS)
    make_model(tmp / "model35", 35)
    model = bk.load_model(tmp / "model35")
    check(model["available"], "fake 35-col model loads")

    # --- case A: subset labels (N=10, auto=[1,4], labels=8) ---
    sdA = tmp / "BLA" / "task" / "sessA"
    XA = make_session(sdA, 10, 8, [1, 4], names35, seed=1, video=True)
    inp = bk.load_session_inputs(sdA)
    check(inp is not None and inp["row0"].tolist() == [0, 2, 3, 5, 6, 7, 8, 9],
          "subset labels -> npz_row0 = complement of auto_rejected")
    res = bk.build_one(sdA, "BLA", "task", "sessA", root, model, P)
    check(res is not None, "build_one writes a key for case A")
    keyA, kpathA = res
    check(kpathA == root / "keys" / "BLA" / "task" / "sessA" / "training_key.mat", "key path layout")
    check(keyA["score_kind"] == "insample" and not np.isnan(keyA["model_score"]).any(),
          "scores present and marked insample")
    expected = model["clf"].predict_proba(model["scaler"].transform(XA))[:, 1][inp["row0"]]
    check(np.allclose(expected, keyA["model_score"]), "scores equal the model's predict_proba on the review rows")
    check(keyA["has_motion_field"] == 1 and keyA["ref_motion"].sum() == 1, "motion field carried")
    check(keyA["ranks_match_npz"] == 1, "recomputed base ranks equal npz columns 14-26")
    check((keyA["contested"] == (keyA["ref_keep"] != keyA["model_keep"]).astype(np.uint8)).all(),
          "contested = ref_keep != model_keep")
    check((keyA["clear_cut"][keyA["contested"] == 1] == 0).all(), "contested items are never clear_cut")
    check(all((c == 9) == bool(k) for c, k in zip(keyA["hint_code"], keyA["contested"])),
          "hint_code 9 exactly on contested items")
    mot_i = int(np.argmax(keyA["ref_motion"]))
    check(("motion artefact" in keyA["hint"][mot_i]), "motion item carries the motion hint text")

    # --- round trip through MATLAB classes ---
    m = sio.loadmat(str(kpathA))
    check(m["ref_keep"].dtype == np.uint8 and m["ref_keep"].shape == (8, 1), "ref_keep uint8 8x1")
    check(m["model_score"].shape == (8, 1) and m["model_score"].dtype == float, "model_score double 8x1")
    check(m["features"].shape == (8, 35) and m["ranks"].shape == (8, 35), "features/ranks 8x35")
    check(m["feature_names"].shape == (1, 35) and str(m["feature_names"][0, 0][0]) == "area",
          "feature_names is a 1x35 cell")
    check(m["hint"].shape == (8, 1) and isinstance(str(m["hint"][0, 0][0]), str), "hint is an 8x1 cell")
    check(m["params"]["CONTEST_SCORE"][0, 0].ravel()[0] == 0.5, "params struct carries CONTEST_SCORE")
    check(str(m["reference_reviewer"][0]) == "Taylor", "reference_reviewer char")
    check(float(m["n_review"].ravel()[0]) == 8 and float(m["n_candidates"].ravel()[0]) == 10, "n_review/n_candidates")

    # --- case B: threshold-0 (labels == N), no motion field ---
    sdB = tmp / "vCA1" / "task" / "sessB"
    make_session(sdB, 10, 10, [], names35, seed=2, with_motion=False)
    inpB = bk.load_session_inputs(sdB)
    check(inpB is not None and inpB["row0"].tolist() == list(range(10)), "labels == N -> identity mapping")
    check(inpB["has_motion"] is False and inpB["motion"].sum() == 0, "missing motion_delete -> zeros")
    keyB, _ = bk.build_one(sdB, "vCA1", "task", "sessB", root, model, P)
    check(keyB["has_motion_field"] == 0, "has_motion_field 0 when the variable is absent")

    # --- case C: inconsistent length (labels = 9 of N=10 with 2 auto) ---
    sdC = tmp / "BLA" / "task" / "sessC"
    make_session(sdC, 10, 9, [1, 4], names35, seed=3)
    check(bk.load_session_inputs(sdC) is None, "unexpected length mismatch is refused")

    # --- case D: 13-column npz refused ---
    sdD = tmp / "BLA" / "task" / "sessD"
    make_session(sdD, 10, 10, [], tc.BASE13, seed=4)
    check(bk.load_session_inputs(sdD) is None, "13-column feature file is refused")

    # --- case E: arity mismatch -> unavailable ---
    make_model(tmp / "model13", 13)
    model13 = bk.load_model(tmp / "model13")
    keyE = bk.make_key(bk.load_session_inputs(sdA), model13, P, "BLA", "task", "sessA")
    check(keyE["score_kind"] == "unavailable" and np.isnan(keyE["model_score"]).all(),
          "13-col model vs 35-col matrix -> score_kind unavailable, NaN scores")
    check(keyE["contested"].sum() == 0 and keyE["clear_cut"].sum() == 0, "no contested/clear flags without scores")
    check(not any(c == 9 for c in keyE["hint_code"]), "no contested hints without scores")

    # --- unsupported area ---
    try:
        tc.check_area("DG_AL"); check(False, "DG_AL refused")
    except ValueError as e:
        check("BLA" in str(e) and "vCA1" in str(e), "DG_AL refused with the supported list")
    check(tc.check_area("bla") == "BLA", "area name is case-insensitive -> canonical")

    # --- v2 hint rules fire on engineered rows ---
    inpF = bk.load_session_inputs(sdB)
    keyF = bk.make_key(inpF, model, P, "vCA1", "task", "sessB")
    col = {n: i for i, n in enumerate(keyF["feature_names"])}
    # force an un-contested delete with top nb_corr_max and bottom ring_contrast
    i = 0
    keyF["ref_keep"][i] = 0; keyF["model_keep"][i] = 0; keyF["contested"][i] = 0
    keyF["ranks"][i, col["nb_corr_max"]] = 1.0
    keyF["ranks"][i, col["ring_contrast"]] = 0.05
    keyF["ranks"][i, col["peak_snr"]] = 0.5; keyF["ranks"][i, col["ev_snr"]] = 0.5
    keyF["ranks"][i, col["area"]] = 0.5
    codes, texts = bk.compute_hints(keyF, P)
    check(codes[i] == 4 and "nb_corr_max" in texts[i] and "ring_contrast" in texts[i],
          "nb_corr_max/ring_contrast rules fire; duplicate takes priority over diffuse")
    keyF["ranks"][i, col["nb_corr_max"]] = 0.5
    codes, texts = bk.compute_hints(keyF, P)
    check(codes[i] == 3 and "surround" in texts[i], "without the duplicate rule the diffuse rule leads")
    # spatial duplicate via the augment fields
    keyF["spatial_ok"] = 1.0
    keyF["overlap_max"] = np.zeros(10); keyF["overlap_partner"] = np.zeros(10)
    keyF["overlap_max"][i] = 0.9; keyF["overlap_partner"][i] = 7
    codes, texts = bk.compute_hints(keyF, P)
    check(codes[i] == 4 and "#7" in texts[i], "spatial overlap duplicate hint names the partner")
    # dim-but-real
    j = int(np.argmax(keyF["ref_keep"] == 1)) if (keyF["ref_keep"] == 1).any() else None
    if j is not None:
        keyF["contested"][j] = 0
        keyF["ranks"][j, col["peak_snr"]] = 0.1
        codes, texts = bk.compute_hints(keyF, P)
        check(codes[j] == 7 and "dim but real" in texts[j], "dim-but-real hint on a low-SNR reference keep")

    # --- spatial merge from a fake MATLAB output ---
    sp = kpathA.parent / tc.SPATIAL_NAME
    sio.savemat(str(sp), {"overlap_max": np.linspace(0, 0.9, 8).reshape(-1, 1),
                          "overlap_partner": np.arange(1, 9).reshape(-1, 1).astype(float),
                          "centroid_yx": np.zeros((8, 2)), "n_pixels": np.ones((8, 1)) * 30,
                          "spatial_ok": 1.0, "augmented_at": "2026-10-01 00:00:00"})
    check(bk.merge_spatial(keyA, sp) and keyA["spatial_ok"] == 1 and keyA["overlap_partner"][7] == 8,
          "spatial merge fills overlap fields")
    keyA["hint_code"], keyA["hint"] = bk.compute_hints(keyA, P)
    bk.save_key(keyA, kpathA)
    m2 = sio.loadmat(str(kpathA))
    check(m2["overlap_max"].shape == (8, 1) and m2["centroid_yx"].shape == (8, 2), "merged key round-trips")

    # --- curriculum row + update ---
    row = bk.curriculum_row(keyA, sdA, kpathA)
    check(row["has_video"] == 1 and row["n_review"] == 8 and row["reviewer"] == "Taylor", "curriculum row fields")
    cpath = bk.update_curriculum(root, [row])
    cpath = bk.update_curriculum(root, [row])   # idempotent
    rows = tc.read_csv_rows(cpath)
    check(len(rows) == 1 and list(rows[0].keys()) == tc.CURRICULUM_COLUMNS, "curriculum.csv has one row, right columns")

    # --- pool filter on the temp tree ---
    pool = tc.pool_sessions("BLA", "taylor", data_parent=tmp)
    check({p.name for p in pool} == {"sessA", "sessC", "sessD"}, "pool filter by area + reviewer (case-insensitive)")
    check(tc.pool_sessions("BLA", "Nobody", data_parent=tmp) == [], "reviewer filter excludes others")

    # --- launcher / marker text ---
    lt = tc.training_launcher_text(Path("C:/repo"))
    check("%" not in lt and lt.isascii() and "acorn_training(session_dir)" in lt, "launcher text ASCII, no %")
    mk = tc.write_training_marker(tmp, {"trainee": "Alice", "session": "BLA/t/s"})
    info = tc.read_training_marker(mk)
    check(info.get("trainee") == "Alice" and "NOTE" not in info, "marker round-trips, NOTE line excluded")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\nRESULT: {'ALL PASS' if not fails else str(len(fails)) + ' FAILURE(S)'}")
sys.exit(1 if fails else 0)
