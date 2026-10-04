"""
Regression test for the MATLAB-column -> numpy-image orientation fix
(features.fcols_to_images; used by the train_classifier retro path and by
features.load_spatial's A.txt fallback).

Run:  C:\\ProgramData\\anaconda3\\envs\\valence\\python.exe test_orientation.py
(also pytest-compatible).  The synthetic checks need no data; the real-session
check runs only when D:\\Julian_CNMFe\\BLA\\.feature_expansion is reachable.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.io as sio

AGENT = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT))
import features  # noqa: E402


def _asym(d1, d2):
    ys, xs = np.mgrid[0:d1, 0:d2]
    return np.exp(-(((ys - 12) / 3.0) ** 2 + ((xs - min(60, d2 - 8)) / 9.0) ** 2))


def test_fcols_to_images_roundtrip_rectangular():
    d1, d2 = 48, 80
    img = _asym(d1, d2)
    A = img.flatten(order="F")[:, None]                 # one MATLAB column
    back = features.fcols_to_images(A, d1, d2)[0]
    assert back.shape == (d1, d2)
    assert np.array_equal(back, img)
    # the old row-major reshape does NOT round-trip (it yields the transpose)
    wrong = A[:, 0].reshape(d1, d2)
    assert np.max(np.abs(wrong - img)) > 0.5


def test_fcols_to_images_square_and_multi():
    d1 = d2 = 64
    imgs = np.stack([_asym(d1, d2), _asym(d1, d2).T])
    A = np.stack([imgs[0].flatten(order="F"), imgs[1].flatten(order="F")], axis=1)
    back = features.fcols_to_images(A, d1, d2)
    assert np.array_equal(back, imgs)
    # sparse input and pixel-count guard
    from scipy import sparse
    assert np.array_equal(features.fcols_to_images(sparse.csc_matrix(A), d1, d2), imgs)
    try:
        features.fcols_to_images(A, d1, d2 + 1)
        assert False, "pixel-count mismatch must raise"
    except ValueError:
        pass


def test_load_spatial_atxt_fallback_uses_cn_dims():
    d1, d2 = 40, 72                                     # non-square frame
    img = _asym(d1, d2)
    with tempfile.TemporaryDirectory() as td:
        sd = Path(td)
        np.savetxt(sd / "A.txt", img.flatten(order="F")[:, None])
        sio.savemat(str(sd / "Cn.mat"), {"Cn": np.zeros((d1, d2))})
        fp = features.load_spatial(sd)
        assert fp.shape == (1, d1, d2)
        assert np.allclose(fp[0], img, atol=1e-6)      # savetxt precision
        (sd / "Cn.mat").unlink()
        try:
            features.load_spatial(sd)
            assert False, "fallback without Cn.mat must refuse"
        except FileNotFoundError:
            pass


def test_cn_correlation_reproduces_stored_column_on_a_real_session():
    """The extraction A of a prospective BLA session, reshaped with the fixed
    helper, must reproduce the stored cn_correlation column bit-for-bit (the
    stored column was computed from spatial_footprints.mat images)."""
    ext = Path(r"D:\Julian_CNMFe\BLA\.feature_expansion\2tones__AVG5x-TSeries-102325-bla8-731um-23z-000.mat")
    sd = Path(r"D:\Julian_CNMFe\BLA\2tones\AVG5x-TSeries-102325-bla8-731um-23z-000")
    if not (ext.exists() and (sd / "candidate_features.npz").exists()):
        print("  (real-session check skipped: data not reachable)")
        return
    m = sio.loadmat(str(ext))
    d1, d2 = int(np.asarray(m["d1"]).flat[0]), int(np.asarray(m["d2"]).flat[0])
    imgs = features.fcols_to_images(m["A"], d1, d2)
    Cn = sio.loadmat(str(sd / "Cn.mat"))["Cn"]
    npz = np.load(sd / "candidate_features.npz", allow_pickle=True)
    reviewed = np.ones(int(npz["n_candidates"][0]), bool)
    reviewed[npz["auto_rejected"].astype(int)] = False
    stored = npz["feature_matrix"][reviewed, 12]
    ours = np.array([features.cn_features(imgs[i], Cn)["cn_correlation"] for i in range(len(imgs))])
    assert len(ours) == len(stored)
    assert np.max(np.abs(ours - stored)) < 1e-9, np.max(np.abs(ours - stored))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"{len(tests)}/{len(tests)} orientation tests pass")
