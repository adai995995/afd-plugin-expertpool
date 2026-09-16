#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Sync a verifiable source snapshot from local Git to the offline shared disk.

Includes tracked and nonignored untracked files (including current edits),
excluding .git. Each invocation creates a new directory, so an ongoing run
keeps its source unchanged. No git, pip, or internet access is needed remotely.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=os.environ.get("EXPERT_POOL_SSH_HOST"),
        help="SSH alias; defaults to EXPERT_POOL_SSH_HOST (no built-in private host)",
    )
    parser.add_argument(
        "--remote-root",
        default=os.environ.get("EXPERT_POOL_REMOTE_ROOT"),
        help="Shared source directory; defaults to EXPERT_POOL_REMOTE_ROOT",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.host or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", args.host):
        parser.error("Set --host or EXPERT_POOL_SSH_HOST to a valid SSH alias")
    if (
        not args.remote_root
        or not PurePosixPath(args.remote_root).is_absolute()
        or any(character in args.remote_root for character in "\n\r\0")
    ):
        parser.error("Set --remote-root or EXPERT_POOL_REMOTE_ROOT to an absolute path")
    root = Path(__file__).resolve().parents[2]
    head = git_output(root, "rev-parse", "HEAD")
    paths = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ]
        )
        .decode()
        .split("\0")
    )
    with tempfile.TemporaryDirectory(prefix="expert-pool-source-") as temp:
        snapshot = Path(temp)
        hashes: dict[str, str] = {}
        for name in sorted(set(paths) - {""}):
            source = root / name
            if source.is_symlink():
                raise ValueError(f"Source snapshot requires regular files: {name}")
            if not source.is_file():
                continue  # A locally deleted tracked file must remain absent.
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
        tree_hash = hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        release = f"{stamp}-{head[:8]}-{tree_hash[:12]}"
        remote = f"{args.remote_root.rstrip('/')}/snapshots/{release}"
        manifest = {
            "git_commit": head,
            "git_branch": git_output(root, "branch", "--show-current"),
            "git_status": git_output(root, "status", "--short"),
            "created_utc": stamp,
            "tree_sha256": tree_hash,
            "files": hashes,
        }
        (snapshot / "source_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        (snapshot / "source_changes.patch").write_text(
            git_output(root, "diff", "--binary", "HEAD") + "\n"
        )
        summary = {
            "host": args.host,
            "remote_source": remote,
            "files": len(hashes),
            "tree_sha256": tree_hash,
        }
        if args.dry_run:
            print(json.dumps({"dry_run": True, **summary}, indent=2))
            return
        ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", args.host]
        prepare = (
            "from pathlib import Path; "
            f"Path({remote!r}).mkdir(parents=True, exist_ok=False)"
        )
        subprocess.run([*ssh, shlex.join(["python3", "-c", prepare])], check=True)
        subprocess.run(
            [
                "rsync",
                "-az",
                "--checksum",
                "-e",
                "ssh -o BatchMode=yes -o ConnectTimeout=12",
                f"{snapshot}/",
                f"{args.host}:{shlex.quote(remote)}/",
            ],
            check=True,
        )
        verify = (
            "import hashlib,json; from pathlib import Path; "
            f"root=Path({remote!r}); "
            "manifest=json.loads((root/'source_manifest.json').read_text()); "
            "bad=[p for p,h in manifest['files'].items() "
            "if hashlib.sha256((root/p).read_bytes()).hexdigest()!=h]; "
            "print(json.dumps({'verified_files':len(manifest['files']),"
            "'mismatches':bad})); "
            "raise SystemExit(bool(bad))"
        )
        subprocess.run([*ssh, shlex.join(["python3", "-c", verify])], check=True)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
