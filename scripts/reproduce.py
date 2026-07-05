#!/usr/bin/env python3
"""PHCA v3.0 — One-command scientific reproduction driver (Phase 19).

Reads reproduce_manifest.json, runs profile steps sequentially, verifies via
existing gate scripts, and writes logs/reproduce_report.json.

Usage:
    python scripts/reproduce.py --profile quick
    python scripts/reproduce.py --profile full
    python scripts/reproduce.py --dry-run --profile quick
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "reproduce_manifest.json"
DEFAULT_REPORT = ROOT / "logs" / "reproduce_report.json"
STDOUT_TAIL_LINES = 20


@dataclass
class StepResult:
    id: str
    name: str
    status: str
    started_at: str
    duration_s: float
    exit_code: int
    verify_pass: bool
    command: str
    outputs: List[str] = field(default_factory=list)
    stdout_tail: List[str] = field(default_factory=list)
    error: Optional[str] = None


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text())
    required = ("manifest_version", "phca_version", "environment", "profiles")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"manifest missing keys: {missing}")
    return data


def validate_manifest(data: dict) -> None:
    for profile_name, profile in data["profiles"].items():
        if "steps" not in profile:
            raise ValueError(f"profile {profile_name!r} missing steps")
        for step in profile["steps"]:
            for key in ("id", "name", "command", "outputs", "verify"):
                if key not in step:
                    raise ValueError(
                        f"profile {profile_name!r} step missing {key!r}: {step.get('id')}"
                    )
            if step["verify"].get("type") not in ("exit_code", "gate_script"):
                raise ValueError(
                    f"step {step['id']!r}: unknown verify type {step['verify'].get('type')!r}"
                )


def build_env(manifest: dict, step: dict) -> dict:
    env = os.environ.copy()
    defaults = manifest.get("environment", {}).get("defaults", {})
    env.update(defaults)
    env.update(step.get("env", {}))
    if "PYTHONPATH" not in step.get("env", {}) and "PYTHONPATH" in defaults:
        existing = os.environ.get("PYTHONPATH", "")
        if existing:
            env["PYTHONPATH"] = f"{defaults['PYTHONPATH']}:{existing}"
    env.setdefault("MUJOCO_GL", "disabled")
    return env


def tail_lines(text: str, n: int = STDOUT_TAIL_LINES) -> List[str]:
    lines = text.rstrip().splitlines()
    return lines[-n:] if len(lines) > n else lines


def run_command(command: str, env: dict, cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, combined, proc.stderr or ""


def run_gate_script(argv: List[str], env: dict, cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, combined


def verify_step(step: dict, command_rc: int, env: dict, cwd: Path) -> tuple[bool, int, str]:
    verify = step["verify"]
    vtype = verify["type"]
    if vtype == "exit_code":
        return command_rc == 0, command_rc, ""
    if vtype == "gate_script":
        if command_rc != 0:
            return False, command_rc, "command failed before gate_script verify"
        argv = verify["argv"]
        gate_rc, gate_out = run_gate_script(argv, env, cwd)
        return gate_rc == 0, gate_rc, gate_out
    return False, 1, f"unknown verify type {vtype!r}"


def check_preflight(manifest: dict) -> tuple[dict, bool]:
    py = sys.version_info
    if py < (3, 11):
        print(
            f"WARNING: Python {py.major}.{py.minor} detected; manifest pins 3.11",
            file=sys.stderr,
        )
    env_info = {
        "python": f"{py.major}.{py.minor}.{py.micro}",
        "mujoco_gl": os.environ.get("MUJOCO_GL", "disabled"),
    }
    reqs = manifest.get("environment", {}).get("requirements", [])
    for req in reqs:
        if not (ROOT / req).is_file():
            print(f"WARNING: requirement file missing: {req}", file=sys.stderr)
    missing: List[str] = []
    try:
        import pytest_timeout  # noqa: F401
    except ImportError:
        missing.append("pytest-timeout")
    try:
        import pytest_benchmark.plugin  # noqa: F401
    except ImportError:
        missing.append("pytest-benchmark")
    if missing:
        print(
            f"ERROR: missing pytest plugins: {', '.join(missing)}",
            file=sys.stderr,
        )
        print(
            "Install: pip install -r requirements.txt -r requirements-dev.txt",
            file=sys.stderr,
        )
        return env_info, False
    return env_info, True


def run_profile(
    manifest: dict,
    profile_name: str,
    *,
    dry_run: bool = False,
) -> tuple[List[StepResult], str]:
    profile = manifest["profiles"][profile_name]
    steps = profile["steps"]
    results: List[StepResult] = []
    overall = "PASS"

    print("=" * 60)
    print(f"  PHCA v3.0 — Reproduce ({profile_name})")
    print(f"  {profile.get('description', '')}")
    est = profile.get("runtime_estimate_s", 0)
    print(f"  Estimated runtime: ~{est // 60} min")
    print("=" * 60)

    for i, step in enumerate(steps, 1):
        step_id = step["id"]
        name = step["name"]
        command = step["command"]
        outputs = step.get("outputs", [])
        started = datetime.now(timezone.utc).isoformat()

        print(f"\n[{i}/{len(steps)}] {step_id}: {name}")
        print(f"  $ {command}")
        if outputs:
            print(f"  → {', '.join(outputs)}")

        if dry_run:
            results.append(StepResult(
                id=step_id,
                name=name,
                status="DRY_RUN",
                started_at=started,
                duration_s=0.0,
                exit_code=0,
                verify_pass=True,
                command=command,
                outputs=outputs,
            ))
            continue

        env = build_env(manifest, step)
        (ROOT / "logs").mkdir(parents=True, exist_ok=True)
        t0 = datetime.now(timezone.utc)
        cmd_rc, combined, _ = run_command(command, env, ROOT)
        verify_ok, verify_rc, verify_out = verify_step(step, cmd_rc, env, ROOT)
        duration = (datetime.now(timezone.utc) - t0).total_seconds()

        all_out = combined
        if verify_out:
            all_out += "\n" + verify_out

        passed = cmd_rc == 0 and verify_ok
        status = "PASS" if passed else "FAIL"
        error = None
        if not passed:
            overall = "FAIL"
            if cmd_rc != 0:
                error = f"command exit {cmd_rc}"
            else:
                error = f"verify exit {verify_rc}"

        result = StepResult(
            id=step_id,
            name=name,
            status=status,
            started_at=started,
            duration_s=round(duration, 2),
            exit_code=cmd_rc if cmd_rc != 0 else verify_rc,
            verify_pass=verify_ok,
            command=command,
            outputs=outputs,
            stdout_tail=tail_lines(all_out),
            error=error,
        )
        results.append(result)
        print(f"  [{status}] {duration:.1f}s")
        if not passed:
            for line in result.stdout_tail[-10:]:
                print(f"    {line}")
            print(f"\n❌ reproduce: FAIL at step {step_id}")
            break

    return results, overall


def write_report(
    manifest: dict,
    profile_name: str,
    results: List[StepResult],
    overall: str,
    env_info: dict,
    output: Path,
    *,
    dry_run: bool = False,
) -> None:
    started = results[0].started_at if results else datetime.now(timezone.utc).isoformat()
    duration = sum(r.duration_s for r in results)
    report = {
        "manifest_version": manifest["manifest_version"],
        "phca_version": manifest["phca_version"],
        "profile": profile_name,
        "dry_run": dry_run,
        "environment": env_info,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(duration, 2),
        "overall": overall if not dry_run else "DRY_RUN",
        "steps": [asdict(r) for r in results],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"\nReport → {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PHCA v3.0 scientific reproduction driver")
    parser.add_argument("--profile", choices=["quick", "full"], default="full")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true", help="List steps without executing")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    validate_manifest(manifest)
    if args.profile not in manifest["profiles"]:
        print(f"Unknown profile: {args.profile}", file=sys.stderr)
        sys.exit(2)

    env_info, preflight_ok = check_preflight(manifest)
    if not preflight_ok and not args.dry_run:
        write_report(
            manifest, args.profile, [], "FAIL", env_info, args.output,
            dry_run=False,
        )
        sys.exit(1)

    results, overall = run_profile(manifest, args.profile, dry_run=args.dry_run)
    write_report(
        manifest, args.profile, results, overall, env_info, args.output,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(f"\n✅ dry-run: {len(results)} steps listed")
        sys.exit(0)

    if overall == "PASS":
        print("\n✅ reproduce: ALL PASS")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
