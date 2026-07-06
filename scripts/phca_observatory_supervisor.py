#!/usr/bin/env python3
"""PHCA Observatory supervisor — subprocess wrapper with crash recovery.

Isolates Qt/cycle crashes from the caller process. On abnormal child exit,
finalizes the partial session (meta + session_report) and optionally runs
phca_replay --check.

Usage:
    PYTHONPATH=python python scripts/phca_observatory_supervisor.py \\
      -- python scripts/phca_observatory.py --cycles=200 --mlp --close-at-end

    QT_QPA_PLATFORM=offscreen PYTHONPATH=python \\
      python scripts/phca_observatory_supervisor.py \\
      --record-dir logs/sessions \\
      -- python scripts/phca_observatory.py --cycles=50 --mlp --close-at-end
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import _bootstrap  # noqa: F401

from phca.monitoring.session_recovery import (  # noqa: E402
    detect_session_state,
    read_latest_pointer,
    recover_session,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _signal_name(returncode: int) -> Optional[str]:
    if returncode >= 0:
        return None
    sig = -returncode
    try:
        return signal.Signals(sig).name
    except (ValueError, AttributeError):
        return f"SIG{sig}"


class SupervisorLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, kind: str, **fields: Any) -> None:
        row = {"ts": _utc_now(), "event": kind, **fields}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")


def _extract_record_dir(observatory_argv: List[str]) -> str:
    for i, tok in enumerate(observatory_argv):
        if tok == "--record-dir" and i + 1 < len(observatory_argv):
            return observatory_argv[i + 1]
        if tok.startswith("--record-dir="):
            return tok.split("=", 1)[1]
    return os.environ.get("PHCA_SESSION_ROOT", "logs/sessions")


def _needs_recovery(session_dir: Optional[Path], child_rc: int) -> bool:
    if child_rc != 0:
        return session_dir is not None
    if session_dir is None:
        return False
    state = detect_session_state(session_dir)
    return state.get("state") in ("running", "incomplete", "corrupt", "empty")


def _build_child_cmd(observatory_argv: List[str]) -> List[str]:
    observatory_script = Path(__file__).resolve().parent / "phca_observatory.py"
    if not observatory_argv:
        return [sys.executable, str(observatory_script)]
    first = observatory_argv[0]
    if Path(first).name in ("python", "python3") or first == sys.executable:
        return list(observatory_argv)
    if Path(first).name == "phca_observatory.py":
        return [sys.executable, *observatory_argv]
    return [sys.executable, str(observatory_script), *observatory_argv]


def run_supervisor(
    observatory_argv: List[str],
    *,
    record_dir: str,
    supervisor_log: Path,
    no_recover: bool = False,
    no_verify: bool = False,
    strict_verify: bool = False,
    preserve_child_exit: bool = False,
) -> int:
    log = SupervisorLog(supervisor_log)
    cmd = _build_child_cmd(observatory_argv)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_pkg_root))

    log.event("supervisor_start", record_dir=record_dir, child_cmd=cmd)
    proc = subprocess.Popen(cmd, env=env)
    log.event("child_spawn", pid=proc.pid, record_dir=record_dir)
    child_rc = int(proc.wait())
    sig = _signal_name(child_rc)
    log.event("child_exit", returncode=child_rc, signal=sig)

    session_dir = read_latest_pointer(record_dir)
    if session_dir is None:
        # Child may have cleared .latest on clean exit; nothing to recover.
        log.event("recover_skip", reason="no_latest_pointer")
        return child_rc

    if no_recover or not _needs_recovery(session_dir, child_rc):
        log.event("recover_skip", session_dir=str(session_dir), child_rc=child_rc)
        return child_rc

    log.event("recover_start", session_dir=str(session_dir), child_rc=child_rc)
    allow_incomplete = not strict_verify
    recover_status = "crashed" if child_rc != 0 else None
    recover_rc, summary = recover_session(
        session_dir,
        write_report=True,
        verify=not no_verify,
        allow_incomplete=allow_incomplete,
        reason="supervisor_crash",
        status=recover_status,
    )
    log.event(
        "recover_done",
        session_dir=str(session_dir),
        status=summary.get("status"),
        jsonl_lines=summary.get("jsonl_lines"),
        expected_cycles=summary.get("expected_cycles"),
        verify_rc=summary.get("verify_rc"),
    )
    if not no_verify:
        log.event(
            "verify_result",
            verify_rc=summary.get("verify_rc"),
            allow_incomplete=allow_incomplete,
            verify_head=(summary.get("verify_status") or "SKIP").splitlines()[0],
        )

    if strict_verify and not no_verify:
        return max(child_rc, recover_rc)
    if preserve_child_exit and child_rc != 0:
        return child_rc
    if child_rc != 0 and recover_rc == 0 and allow_incomplete:
        return 0
    return child_rc if child_rc != 0 else recover_rc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supervise phca_observatory.py in a subprocess with crash recovery",
    )
    parser.add_argument(
        "--supervisor-log",
        default=None,
        help="JSONL supervisor log path (default: logs/supervisor/<ts>.jsonl)",
    )
    parser.add_argument(
        "--record-dir",
        default=None,
        help="session record root (default: parse from child argv or logs/sessions)",
    )
    parser.add_argument("--no-recover", action="store_true",
                        help="do not finalize partial sessions on abnormal exit")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip phca_replay --check after recovery")
    parser.add_argument("--strict-verify", action="store_true",
                        help="fail if strict --check fails after recovery")
    parser.add_argument(
        "--preserve-child-exit",
        action="store_true",
        help="return child exit code even when recovery and verify succeed",
    )
    parser.add_argument(
        "observatory_args",
        nargs=argparse.REMAINDER,
        help="arguments after -- forwarded to phca_observatory.py",
    )
    args = parser.parse_args()

    obs_argv = list(args.observatory_args)
    if obs_argv and obs_argv[0] == "--":
        obs_argv = obs_argv[1:]

    record_dir = args.record_dir or _extract_record_dir(obs_argv)
    if args.supervisor_log:
        log_path = Path(args.supervisor_log)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path("logs/supervisor") / f"{ts}.jsonl"

    rc = run_supervisor(
        obs_argv,
        record_dir=record_dir,
        supervisor_log=log_path,
        no_recover=args.no_recover,
        no_verify=args.no_verify,
        strict_verify=args.strict_verify,
        preserve_child_exit=args.preserve_child_exit,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
