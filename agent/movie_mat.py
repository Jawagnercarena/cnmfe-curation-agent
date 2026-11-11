"""
movie_mat.py -- a session's CNMF-E movie (<session>/<session>.mat) and its source .tif.

<session>.mat holds Y (d1 x d2 x T) and Ysiz.  tif2mat writes it at the start of
every headless run, as a frame-exact re-encoding of <session>.tif (14 sessions
sampled 2026-11-11, BLA/vCA1/DG_AL, uint8 and uint16, 2022-2026).  Once a session
is finished it is pruned (prune_movie_mats.py), and the tools that ship it to
somebody rebuild it here first:
    push_review_bundle.collect_files  (the push, and rereview.py request):
                                      into the session folder
    push_training_bundle.stage_one    into DATA_PARENT/.training/movies/<area>/<task>/
                                      <session>/ (training never writes a production
                                      folder); prune_movie_mats.py clears that cache too
CNMFe_final_save.m rebuilds it itself, into the session folder.

The rebuild runs ca_source_extraction/endoscope/rebuild_movie_mat.m in MATLAB:
it writes under <session>/.movie_rebuild/, checks the result, and moves it into
place only when complete, so an interrupted rebuild never leaves a truncated
movie under the real name.  It never overwrites an existing movie.  Python then
checks the rebuilt file against the .tif with an independent reader (tifffile).

Local only: everything here reads or writes inside one local session folder.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

REBUILD_TIMEOUT_H = 1.5     # longest logged tif -> mat conversion: 1268 s
VERIFY_FRAMES = 32          # sampled frames, first and last always included
_FULL_CHUNK = 256           # frames per block for a full comparison


def movie_mat(session_dir: Path) -> Path:
    session_dir = Path(session_dir)
    return session_dir / f"{session_dir.name}.mat"


def movie_tif(session_dir: Path) -> Path | None:
    """The session's source .tif (exact folder name, .tif or .tiff), or None."""
    session_dir = Path(session_dir)
    for ext in (".tif", ".tiff"):
        p = session_dir / f"{session_dir.name}{ext}"
        if p.is_file():
            return p
    return None


class _TifFrames:
    """Frame access to a tif, oriented (T, d1, d2) like the tif pages.
    Memory-maps when the file allows it (uncompressed, contiguous: covers the
    >4 GB single-IFD ImageJ files), else reads pages through the series."""

    def __init__(self, tif: Path):
        import tifffile
        self._tf = None
        self._mm = None
        try:
            self._mm = tifffile.memmap(str(tif), mode="r")
            self.shape = tuple(self._mm.shape)
            self.dtype = self._mm.dtype
        except Exception:
            self._mm = None
            self._tf = tifffile.TiffFile(str(tif))
            s = self._tf.series[0]
            self._series = s
            self.shape = tuple(s.shape)
            self.dtype = s.dtype

    def frames(self, start: int, stop: int) -> np.ndarray:
        """Frames [start, stop) as an in-memory copy, shape (n, d1, d2)."""
        if self._mm is not None:
            out = np.array(self._mm[start:stop])        # copy: the map is closed later
        elif len(self.shape) == 2:
            out = np.array(self._series.asarray())
        else:
            out = np.array(self._series.asarray(key=slice(start, stop)))
        return out[None] if out.ndim == 2 else out

    def close(self):
        if self._mm is not None:
            mm = self._mm
            self._mm = None
            if getattr(mm, "_mmap", None) is not None:
                mm._mmap.close()
            del mm
        if self._tf is not None:
            self._tf.close()
            self._tf = None


def verify_against_tif(mat: Path, tif: Path, n_frames: int = VERIFY_FRAMES,
                       full: bool = False) -> tuple[bool, str]:
    """Does the movie .mat hold exactly the tif's frames?

    Checks Y's shape and Ysiz against the tif's (T, d1, d2), the pixel type, and
    n_frames evenly spaced frames (or, with full=True, every frame).  MATLAB writes
    Y as (d1, d2, T); h5py exposes it as (T, d2, d1), so an h5 frame is the
    transpose of a tif page.  Returns (ok, detail); never raises on bad input."""
    import h5py
    reader = None
    try:
        reader = _TifFrames(tif)
        if len(reader.shape) != 3:
            return False, f"tif is not a single-channel stack (shape {reader.shape})"
        Tt, t1, t2 = reader.shape
        with h5py.File(str(mat), "r") as h:
            if "Y" not in h or "Ysiz" not in h:
                return False, "movie has no Y / Ysiz"
            Y = h["Y"]
            if Y.ndim != 3:
                return False, f"movie Y is {Y.ndim}-D"
            T, d2, d1 = (int(s) for s in Y.shape)
            ysiz = [int(round(float(v))) for v in np.asarray(h["Ysiz"]).ravel()]
            mdt = np.dtype(Y.dtype)
            if ysiz != [d1, d2, T]:
                return False, f"Ysiz {ysiz} disagrees with Y ({d1} x {d2} x {T})"
            if (Tt, t1, t2) != (T, d1, d2):
                return False, f"tif is {t1} x {t2} x {Tt}, movie is {d1} x {d2} x {T}"
            tdt = np.dtype(reader.dtype)
            if (tdt.kind, tdt.itemsize) != (mdt.kind, mdt.itemsize):
                return False, f"pixel type differs (tif {tdt}, movie {mdt})"
            if full:
                for s in range(0, T, _FULL_CHUNK):
                    e = min(s + _FULL_CHUNK, T)
                    a = reader.frames(s, e)
                    b = np.asarray(Y[s:e]).transpose(0, 2, 1)
                    if not np.array_equal(a, b):
                        bad = s + int(np.flatnonzero(
                            ~(a == b).reshape(e - s, -1).all(axis=1))[0])
                        return False, f"frame {bad} differs"
                return True, f"{d1} x {d2} x {T} {mdt}, all {T} frames equal"
            idx = sorted(set(np.linspace(0, T - 1, min(n_frames, T)).round().astype(int).tolist()))
            for i in idx:
                a = reader.frames(i, i + 1)[0]
                b = np.asarray(Y[i]).T
                if not np.array_equal(a, b):
                    return False, f"frame {i} differs"
            return True, f"{d1} x {d2} x {T} {mdt}, {len(idx)}/{len(idx)} sampled frames equal"
    except Exception as e:      # unreadable / truncated / not HDF5: not verified
        return False, f"{type(e).__name__}: {e}"
    finally:
        if reader is not None:
            reader.close()


def _matlab_str(p: Path) -> str:
    s = Path(p).as_posix()
    if "%" in s:
        # _run_matlab joins the script onto one -batch line, where % starts a comment
        raise ValueError(f"cannot pass a path containing '%' to MATLAB -batch: {s}")
    return s.replace("'", "''")


def _target(session_dir: Path, out_dir: Path | None) -> Path:
    return movie_mat(session_dir) if out_dir is None else Path(out_dir) / movie_mat(session_dir).name


def rebuild(session_dir: Path, log=print, out_dir: Path | None = None) -> Path:
    """Rebuild <session>.mat from <session>.tif in MATLAB, then check it against
    the tif.  The movie goes into the session folder, or into out_dir when given
    (the training push keeps it out of the production folder).  Raises
    (FileNotFoundError / RuntimeError) on any failure; returns the movie path on
    success.  An existing movie is returned untouched."""
    from local_config import REPO_ROOT
    import run_cnmfe

    mat, tif = _target(session_dir, out_dir), movie_tif(session_dir)
    if mat.exists():
        return mat
    if tif is None:
        raise FileNotFoundError(f"no {Path(session_dir).name}.tif to rebuild the movie from")
    mat.parent.mkdir(parents=True, exist_ok=True)
    lib = Path(REPO_ROOT) / "ca_source_extraction"
    script = (f"addpath(genpath('{_matlab_str(lib)}')); "
              f"rebuild_movie_mat('{_matlab_str(tif)}', '{_matlab_str(mat)}');")
    ok = run_cnmfe._run_matlab(script, log, timeout_hours=REBUILD_TIMEOUT_H)
    if not ok or not mat.exists():
        raise RuntimeError(f"rebuilding {mat.name} from {tif.name} failed (MATLAB log above); "
                           f"a partial copy, if any, is under {mat.parent / '.movie_rebuild'}")
    good, detail = verify_against_tif(mat, tif)
    if not good:
        raise RuntimeError(f"rebuilt {mat.name} does not match {tif.name} ({detail}); "
                           f"not using it -- move it aside by hand before trying again")
    log(f"  rebuilt {mat.name} from {tif.name}: {detail}")
    return mat


def ensure(session_dir: Path, dry_run: bool = False, log=print,
           out_dir: Path | None = None) -> Path | None:
    """The session's movie .mat, rebuilt from its .tif first when it is missing.

    The session folder's own movie is used whenever it exists.  Otherwise the
    movie is rebuilt into the session folder, or into out_dir when given (and an
    earlier rebuild already there is reused).  Returns the movie path when it
    exists or was rebuilt.  Returns None when there is no movie and no .tif to
    rebuild it from, and on a dry run that would rebuild (it says so; nothing is
    run).  A failed rebuild raises."""
    own = movie_mat(session_dir)
    if own.exists():
        return own
    mat = _target(session_dir, out_dir)
    if mat.exists():
        log(f"  using the rebuilt movie at {mat}")
        return mat
    tif = movie_tif(session_dir)
    if tif is None:
        return None
    where = "" if out_dir is None else f" into {Path(out_dir)}"
    if dry_run:
        log(f"  would rebuild {mat.name} from {tif.name}{where} ({tif.stat().st_size / 1e9:.1f} GB tif, "
            f"usually 1-3 min; not counted in the sizes below)")
        return None
    log(f"  {own.name} not found; rebuilding it from {tif.name}{where} (usually 1-3 min) ...")
    return rebuild(session_dir, log, out_dir=out_dir)
