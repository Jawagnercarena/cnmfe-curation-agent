"""
run_cnmfe.py
Launches MATLAB headless to run CNMFe_Biane_headless.m on a session.
Also handles the pre-run step that generates Cn.mat + pnr.mat for parameter
estimation before the full run.

cnmfe_setup.m is called first in every MATLAB invocation — it handles all
path setup (ca_source_extraction, cvx, deconvolveCa, GUI) the same way
the interactive pipeline does.
"""

import subprocess
import threading
import time
from pathlib import Path

from local_config import MATLAB_EXE, REPO_ROOT as _REPO_ROOT
REPO_ROOT = str(_REPO_ROOT)

# Quick MATLAB pass: compute and save Cn.mat + pnr.mat only.
# Python reads these to estimate min_corr, min_pnr, bd before the full run.
_PRE_RUN_SCRIPT = """
cd('{repo}');
run('cnmfe_setup.m');
global d1 d2 numFrame ssub tsub sframe num2read Fs neuron neuron_ds neuron_full Ybg_weights;
nam = '{nam}';
cnmfe_choose_data;
Fs = 3.75; ssub = 2; tsub = 2; gSig = {gSig}; gSiz = {gSiz};
neuron_full = Sources2D('d1',d1,'d2',d2,'ssub',ssub,'tsub',tsub,'name',nam_mat,'gSig',gSig,'gSiz',gSiz);
neuron_full.Fs = Fs;
neuron_full.options.search_method = 'ellipse';
neuron_full.options.dist = 3;
sframe = 1; num2read = numFrame;
cnmfe_load_data;
Y = neuron.reshape(Y, 1);
[Cn, pnr] = neuron.correlation_pnr(Y(:, round(linspace(1, T, min(T, 1000)))));
cd('{session_dir}');
save('Cn.mat', 'Cn');
save('pnr.mat', 'pnr');
fprintf('Pre-run complete. Cn and pnr saved.\\n');
"""

# Full headless CNMFe run.
_FULL_RUN_SCRIPT = """
cd('{repo}');
run('cnmfe_setup.m');
nam = '{nam}';
gSig = {gSig}; gSiz = {gSiz};
min_corr = {min_corr}; min_pnr = {min_pnr}; bd = {bd};
run('CNMFe_Biane_headless.m');
"""


def _run_matlab(script: str, log, timeout_hours: float = 3.0) -> bool:
    """
    Run an inline MATLAB script via -batch. Returns True on success.
    Streams MATLAB stdout in real-time so progress is visible immediately.
    """
    # Collapse multi-line script to single line for -batch
    single_line = " ".join(script.strip().splitlines())
    cmd = [MATLAB_EXE, "-batch", single_line]

    log(f"  [MATLAB] Starting MATLAB...")
    start = time.time()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Stream stdout in real-time from a background thread
        stdout_lines = []
        def _read_stdout():
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    log(f"  [MATLAB] {line}")
                    stdout_lines.append(line)

        reader = threading.Thread(target=_read_stdout, daemon=True)
        reader.start()

        try:
            proc.wait(timeout=timeout_hours * 3600)
        except subprocess.TimeoutExpired:
            proc.kill()
            reader.join(timeout=5)
            log(f"  [MATLAB] TIMEOUT after {timeout_hours} hours.")
            return False

        reader.join(timeout=10)
        elapsed = time.time() - start

        if proc.returncode == 0:
            log(f"  [MATLAB] Finished successfully in {elapsed/60:.1f} min.")
            return True
        else:
            stderr = proc.stderr.read()
            log(f"  [MATLAB] ERROR (returncode={proc.returncode}) "
                f"after {elapsed/60:.1f} min.")
            if stderr.strip():
                log(f"  [MATLAB] stderr tail:\n{stderr[-1000:]}")
            return False

    except Exception as e:
        log(f"  [MATLAB] Unexpected error: {e}")
        return False


def run_pre_pass(session_name: str, session_dir: Path, gSig: int, gSiz: int, log) -> bool:
    """
    Quick MATLAB pass (~1-2 min) to compute Cn.mat and pnr.mat.
    Python reads these to estimate min_corr, min_pnr, and bd.

    We pass the .tif path to cnmfe_choose_data, which converts it to .mat
    automatically and sets nam_mat. On subsequent calls (full run) the .mat
    already exists and conversion is skipped.
    """
    nam = str(session_dir / f"{session_name}.tif").replace("\\", "/")
    sd  = str(session_dir).replace("\\", "/")

    script = _PRE_RUN_SCRIPT.format(
        repo=REPO_ROOT.replace("\\", "/"),
        nam=nam,
        gSig=gSig,
        gSiz=gSiz,
        session_dir=sd,
    )

    log(f"\n[RUN] Pre-pass: computing Cn and pnr images...")
    return _run_matlab(script, log, timeout_hours=0.5)


def run_pre_pass_to_dir(tif_path: Path, save_dir: Path, gSig: int, gSiz: int, log) -> bool:
    """
    Quick MATLAB pass (~2 min) to compute Cn.mat and pnr.mat, saving them to
    save_dir instead of the session folder.  Used by bootstrap_preagent.py for
    pre-agent sessions that don't have pnr.mat yet.
    """
    nam = str(tif_path).replace("\\", "/")
    sd  = str(save_dir).replace("\\", "/")

    script = _PRE_RUN_SCRIPT.format(
        repo=REPO_ROOT.replace("\\", "/"),
        nam=nam,
        gSig=gSig,
        gSiz=gSiz,
        session_dir=sd,
    )

    log(f"\n[RUN] Pre-pass (bootstrap): computing Cn and pnr for param estimation...")
    return _run_matlab(script, log, timeout_hours=0.5)


def run_full_cnmfe(session_name: str, session_dir: Path, params: dict, log) -> bool:
    """
    Full headless CNMFe run with the estimated parameters.
    Saves all standard outputs plus Ybg.mat and pnr.mat to session_dir.

    We pass the .tif path. cnmfe_choose_data will find the already-converted
    .mat file (created during the pre-pass) and skip re-conversion.
    """
    nam = str(session_dir / f"{session_name}.tif").replace("\\", "/")

    script = _FULL_RUN_SCRIPT.format(
        repo=REPO_ROOT.replace("\\", "/"),
        nam=nam,
        gSig=params["gSig"],
        gSiz=params["gSiz"],
        min_corr=params["min_corr"],
        min_pnr=params["min_pnr"],
        bd=params["bd"],
    )

    log(f"\n[RUN] Full CNMFe headless run with params: {params}")
    return _run_matlab(script, log, timeout_hours=4.0)
