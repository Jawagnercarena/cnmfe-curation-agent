"""
assemble_report.py -- inject the synthesized tables into redteam_report.md.

synth_report.py prints Markdown blocks separated by '## ' headings; this
replaces the <!-- SYNTH:... --> placeholders in the report with them, so every
table in the report is generated from results/*.json.  Idempotent: a block is
re-injected between its placeholder and the matching <!-- /SYNTH:... --> end
marker if one exists.
"""
import re
import subprocess
import sys

import rt_lib as rt

REPORT = rt.SP / "redteam_report.md"


def synth_blocks():
    out = subprocess.run([sys.executable, str(rt.SP / "synth_report.py")], cwd=str(rt.SP),
                         capture_output=True, text=True, check=True).stdout
    header, rest = out.split("\n## ", 1)
    blocks = {"HEADER": header.strip()}
    for chunk in ("## " + rest).split("\n## "):
        chunk = chunk if chunk.startswith("## ") else "## " + chunk
        title = chunk.splitlines()[0][3:].strip()
        body = "\n".join(chunk.splitlines()[1:]).strip()
        key = {"Verdict table": "VERDICT_TABLE", "Section-D scoreboard": "SCOREBOARD",
               "Deploy-verdict inputs": "DEPLOY_TABLE"}.get(title)
        if key:
            blocks[key] = body
    return blocks


def inject(text, key, body):
    start = f"<!-- SYNTH:{key} -->"
    end = f"<!-- /SYNTH:{key} -->"
    if start not in text:
        return text
    pat = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    replacement = f"{start}\n{body}\n{end}"
    if pat.search(text):
        return pat.sub(lambda m: replacement, text)
    return text.replace(start, replacement)


def main():
    text = REPORT.read_text(encoding="utf-8")
    for key, body in synth_blocks().items():
        text = inject(text, key, body)
    cc = rt.RESULTS / "closing_check.json"
    if cc.exists():
        import json
        c = json.loads(cc.read_text())
        body = (f"Closing check: pin unchanged = {c['pin_unchanged']}; {c['files_scanned']} files scanned under the three area "
                f"roots + `.bootstrap_diag`, {c['n_files_newer_than_pin']} newer than the pin; tracked files modified: "
                f"{c['tracked_modified'] or 'none'}; unexpected untracked: {c['unexpected_untracked'] or 'none'} -> "
                f"**{'PASS' if c['ok'] else 'FAIL'}**.")
        text = inject(text, "CLOSING", body)
    REPORT.write_text(text, encoding="utf-8")
    print("assembled", REPORT.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
