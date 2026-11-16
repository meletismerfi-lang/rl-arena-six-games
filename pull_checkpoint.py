"""Pull a file off the RunPod pod through the shell-only SSH proxy, with SHA-256 verification.

The proxy has no sftp/scp, so the file is base64'd on the pod, streamed through stdin-driven
bash, and reassembled locally. An unguessable marker wraps the payload so the echoed command
line and the login banner cannot contaminate the data.
"""
from __future__ import annotations

import base64
import hashlib
import re
import subprocess
import sys
from pathlib import Path

HOST = "b7s1fdx2cjn2hq-64412a3b@ssh.runpod.io"
KEY = str(Path.home() / ".ssh" / "id_ed25519")
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def run_remote(commands: str, timeout: int = 900) -> str:
    proc = subprocess.run(
        ["ssh", "-tt", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
         "-o", "ConnectTimeout=20", "-i", KEY, HOST],
        input=f"{commands}\nsleep 3\nexit\n",
        capture_output=True, text=True, timeout=timeout, errors="replace",
    )
    out = ANSI.sub("", proc.stdout).replace("\r", "")
    return out


def sha_remote(path: str) -> str:
    out = run_remote(f"sha256sum {path}")
    m = re.search(r"([0-9a-f]{64})", out)
    if not m:
        raise SystemExit(f"could not read remote sha256. output tail:\n{out[-500:]}")
    return m.group(1)


def pull(path: str, dest: Path) -> None:
    print(f"remote sha256: {sha_remote(path)}")
    marker = base64.b16encode(hashlib.sha256(str(Path(dest)).encode()).digest())[:12].decode()
    cmds = (f'base64 -w 76 {path} | sed "s/^/A{marker}/" ; echo "B{marker}"')
    print("streaming base64 ...", flush=True)
    out = run_remote(cmds, timeout=1800)
    chunks = []
    for line in out.splitlines():
        if line.startswith(f"A{marker}"):
            chunks.append(line[13:])
        elif line.startswith(f"B{marker}"):
            break
    payload = "".join(chunks)
    if not payload:
        raise SystemExit(f"no payload captured. output tail:\n{out[-400:]}")
    data = base64.b64decode(payload, validate=False)
    dest.write_bytes(data)
    local_sha = hashlib.sha256(data).hexdigest()
    remote_sha = sha_remote(path)
    print(f"local  sha256: {local_sha}")
    print(f"bytes: {len(data):,}")
    print("MATCH" if local_sha == remote_sha else "MISMATCH")


if __name__ == "__main__":
    remote_path = sys.argv[1]
    local_path = Path(sys.argv[2])
    local_path.parent.mkdir(parents=True, exist_ok=True)
    pull(remote_path, local_path)
