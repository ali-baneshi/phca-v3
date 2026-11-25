"""Qt-free session recovery and finalization after abnormal Observatory exits."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from phca.monitoring.observability import normalize_observability_json

PathLike = Union[str, Path]

TERMINAL_STATUSES = frozenset({"complete", "incomplete", "empty", "crashed"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_latest_pointer(record_root: PathLike) -> Optional[Path]:
    """Return session dir from ``<record_root>/.latest`` if present."""
    p = Path(record_root) / ".latest"
    if not p.is_file():
        return None
    try:
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            return None
        return Path(text)
    except Exception:
        return None


def count_jsonl_lines(session_dir: PathLike) -> int:
    jsonl_p = Path(session_dir) / "timeseries.jsonl"
    if not jsonl_p.exists():
        return 0
    n = 0
    with jsonl_p.open(encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                n += 1
    return n


def _load_meta(session_dir: Path) -> Optional[Dict[str, Any]]:
    meta_p = session_dir / "meta.json"
    if not meta_p.exists():
        return None
    try:
        return json.loads(meta_p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _scan_jsonl(session_dir: Path) -> Tuple[int, int, List[int]]:
    """Return (good_lines, bad_lines, bad_line_numbers)."""
    jsonl_p = session_dir / "timeseries.jsonl"
    if not jsonl_p.exists():
        return 0, 0, []
    good = 0
    bad = 0
    bad_nums: List[int] = []
    with jsonl_p.open(encoding="utf-8") as fh:
        for i, ln in enumerate(fh, start=1):
            if not ln.strip():
                continue
            try:
                normalize_observability_json(json.loads(ln))
                good += 1
            except Exception:
                bad += 1
                bad_nums.append(i)
    return good, bad, bad_nums


def detect_session_state(session_dir: PathLike) -> Dict[str, Any]:
    """Classify session integrity for recovery decisions."""
    d = Path(session_dir)
    meta = _load_meta(d)
    if meta is None:
        return {
            "session_dir": str(d),
            "state": "missing",
            "jsonl_lines": 0,
            "expected_cycles": None,
            "recorded_cycles": None,
            "meta_status": None,
            "corrupt_lines": 0,
        }

    good, bad, bad_nums = _scan_jsonl(d)
    expected_raw = meta.get("cycles")
    expected = int(expected_raw) if isinstance(expected_raw, (int, float)) else None
    recorded_raw = meta.get("recorded_cycles")
    recorded = int(recorded_raw) if isinstance(recorded_raw, (int, float)) else None
    meta_status = meta.get("status")

    if bad > 0:
        state = "corrupt"
    elif good == 0:
        state = "empty"
    elif meta_status == "complete" and recorded is not None and good == recorded:
        if expected is None or good == expected:
            state = "complete"
        else:
            state = "incomplete"
    elif meta_status == "running" or recorded is None:
        state = "running" if good > 0 else "empty"
    elif expected is not None and good < expected:
        state = "incomplete"
    elif recorded is not None and good != recorded:
        state = "incomplete"
    elif meta_status in TERMINAL_STATUSES:
        state = str(meta_status)
    else:
        state = "incomplete"

    return {
        "session_dir": str(d),
        "state": state,
        "jsonl_lines": good,
        "expected_cycles": expected,
        "recorded_cycles": recorded,
        "meta_status": meta_status,
        "corrupt_lines": bad,
        "corrupt_line_numbers": bad_nums[:10],
    }


def finalize_session(
    session_dir: PathLike,
    *,
    reason: str = "recover",
    write_report: bool = True,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """Sync meta from JSONL ground truth and optionally write session_report.json."""
    d = Path(session_dir)
    meta = _load_meta(d)
    if meta is None:
        raise FileNotFoundError(f"not a valid session dir: {d}")

    good, bad, _ = _scan_jsonl(d)
    expected_raw = meta.get("cycles")
    expected = int(expected_raw) if isinstance(expected_raw, (int, float)) else None

    if status is None:
        if bad > 0:
            status = "corrupt"
        elif good == 0:
            status = "empty"
        elif expected is not None and good < expected:
            status = "incomplete"
        elif expected is not None and good == expected:
            status = "complete"
        else:
            status = "incomplete"

    meta["status"] = status
    meta["recorded_cycles"] = good
    meta["closed_at"] = _utc_now()
    meta["recovery_reason"] = reason
    (d / "meta.json").write_text(json.dumps(meta, indent=2))

    report_path: Optional[Path] = None
    report_error: Optional[str] = None
    if write_report and good > 0:
        try:
            from phca.monitoring.session_report import write_session_report
            write_session_report(d)
            report_path = d / "session_report.json"
        except Exception as exc:
            report_error = str(exc)

    latest_p = d.parent / ".latest"
    try:
        if latest_p.exists() and latest_p.read_text(encoding="utf-8").strip() == str(d.resolve()):
            latest_p.unlink()
    except Exception:
        pass

    return {
        "session_dir": str(d),
        "status": status,
        "jsonl_lines": good,
        "expected_cycles": expected,
        "corrupt_lines": bad,
        "recovery_reason": reason,
        "report_path": str(report_path) if report_path else None,
        "report_error": report_error,
    }


def _run_replay_check(
    session_dir: Path,
    *,
    allow_incomplete: bool = False,
) -> Tuple[int, str]:
    repo = Path(__file__).resolve().parents[3]
    replay = repo / "scripts" / "phca_replay.py"
    env = dict(__import__("os").environ)
    env.setdefault("PYTHONPATH", str(repo / "python"))
    cmd = [sys.executable, str(replay), "--check", str(session_dir)]
    if allow_incomplete:
        cmd.append("--allow-incomplete")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")
    status = "PASS" if proc.returncode == 0 else "FAIL"
    return int(proc.returncode), f"{status}\n{output}".strip()


def recover_session(
    session_dir: PathLike,
    *,
    write_report: bool = True,
    verify: bool = False,
    allow_incomplete: bool = False,
    reason: str = "recover",
    status: Optional[str] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Finalize a partial session and optionally run phca_replay --check."""
    summary = finalize_session(
        session_dir, reason=reason, write_report=write_report, status=status)
    verify_rc = 0
    verify_status = "SKIP"
    if verify:
        verify_rc, verify_status = _run_replay_check(
            Path(session_dir), allow_incomplete=allow_incomplete)
        summary["verify_status"] = verify_status
        summary["verify_rc"] = verify_rc
    else:
        summary["verify_status"] = verify_status
        summary["verify_rc"] = verify_rc

    exit_code = verify_rc if verify else 0
    summary["exit_code"] = exit_code
    return exit_code, summary


def print_recover_summary(summary: Dict[str, Any], *, stream=None) -> None:
    out = stream or sys.stdout
    print("=== Session recover ===", file=out)
    print(f"  dir:      {summary.get('session_dir')}", file=out)
    print(f"  status:   {summary.get('status')}", file=out)
    print(f"  lines:    {summary.get('jsonl_lines')} / {summary.get('expected_cycles')}", file=out)
    if summary.get("corrupt_lines"):
        print(f"  corrupt:  {summary.get('corrupt_lines')} line(s)", file=out)
    if summary.get("report_path"):
        print(f"  report:   {summary.get('report_path')}", file=out)
    elif summary.get("report_error"):
        print(f"  report:   FAIL ({summary.get('report_error')})", file=out)
    if summary.get("verify_status") and summary.get("verify_status") != "SKIP":
        print(f"  verify:   {summary.get('verify_status', '').splitlines()[0]}", file=out)
    print(f"=== Exit {summary.get('exit_code', 0)} ===", file=out)
