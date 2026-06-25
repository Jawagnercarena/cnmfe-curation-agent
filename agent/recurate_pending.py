"""
recurate_pending.py
Re-runs curator.prepare_review_package() on all sessions that have been
processed by the headless run (ROIs_candidates.jpg exists) but not yet
reviewed by the user (no ROIs.jpg).

Run this once after training the binary classifier so the review queue
benefits from the new model's more aggressive rejection.

Usage:
    cd C:\code\CNMF_E_LEGACY_BIANE_CLAUDE\agent
    python recurate_pending.py
"""

import sys
from pathlib import Path

AGENT_DIR = Path(__file__).parent
from config import DATA_ROOT

sys.path.insert(0, str(AGENT_DIR))
import curator


def log(msg: str):
    print(msg, flush=True)


def find_pending_sessions() -> list[tuple[str, Path]]:
    """Sessions with ROIs_candidates.jpg but no ROIs.jpg."""
    pending = []
    for task_dir in sorted(DATA_ROOT.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        for session_dir in sorted(task_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            if ((session_dir / "ROIs_candidates.jpg").exists() and
                    not (session_dir / "ROIs.jpg").exists()):
                session_name = session_dir.name
                pending.append((session_name, session_dir))
    return pending


def main():
    pending = find_pending_sessions()
    log(f"Found {len(pending)} pending sessions to re-curate.\n")

    if not pending:
        log("Nothing to do.")
        return

    for i, (session_name, session_dir) in enumerate(pending, 1):
        log(f"{'=' * 60}")
        log(f"[{i}/{len(pending)}] {session_dir.parent.name}/{session_name}")
        log(f"{'=' * 60}")
        try:
            curator.prepare_review_package(session_name, session_dir, log)
        except Exception as e:
            log(f"  ERROR: {e}")
            import traceback
            log(traceback.format_exc())

    log(f"\nDone. {len(pending)} sessions re-curated with the new classifier.")
    log("Open MATLAB and run run_final_review.m for each session to start reviewing.")


if __name__ == "__main__":
    main()
