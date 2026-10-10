"""Network access evidence for data sources (manifests/network_requirements.json).

Repeats each recorded operation through the workspace's egress proxy and records
the outcome: reachable (HTTP status from the destination), blocked by the egress
policy (the proxy refuses the CONNECT with 403), DNS failure, or other error.
Small range requests only; nothing is stored.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..utils import utc_now, write_json


def probe_one(op: str, timeout: int = 25) -> dict:
    method, url = op.split(" ", 1)
    url = url.split(" ")[0]
    cmd = ["curl", "-sS", "-o", "/dev/null", "--max-time", str(timeout), "-w", "%{http_code}"]
    cmd += ["-I"] if method == "HEAD" else ["-r", "0-1023"]
    r = subprocess.run(cmd + [url], capture_output=True, text=True)
    err = r.stderr.strip()
    if "CONNECT tunnel failed, response 403" in err:
        state = "blocked_by_egress_policy"
    elif "Could not resolve host" in err:
        state = "dns_failure"
    elif r.returncode == 0:
        state = "reachable"
    else:
        state = "error"
    return {"state": state, "http_status": r.stdout.strip() or None, "curl_exit": r.returncode, "error": err[:200] or None}


def probe(manifest: Path, out: Path) -> dict:
    req = json.loads(manifest.read_text())
    rows = []
    for h in req["hosts"]:
        res = probe_one(h["operation"])
        rows.append({"host": h["host"], "operation": h["operation"], **res, "unblocks": h["unblocks"], "resume": h["resume"]})
    summary = {"checked_at": utc_now().isoformat(timespec="seconds"), "via": "workspace egress proxy (HTTPS_PROXY)",
               "blocked": [r["host"] for r in rows if r["state"] != "reachable"], "reachable": [r["host"] for r in rows if r["state"] == "reachable"],
               "hosts": rows}
    write_json(out, summary)
    return summary
