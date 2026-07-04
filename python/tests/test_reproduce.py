"""Tests for PHCA v3.0 scientific reproduction manifest and driver."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "reproduce_manifest.json"
REPRODUCE_SCRIPT = ROOT / "scripts" / "reproduce.py"


def _load_reproduce_module():
    spec = importlib.util.spec_from_file_location("reproduce", REPRODUCE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_manifest_schema_valid():
    data = json.loads(MANIFEST.read_text())
    assert data["manifest_version"] == 1
    assert data["phca_version"] == "3.0"
    assert "environment" in data
    assert data["environment"]["python"] == "3.11"
    assert data["environment"]["mujoco_gl"] == "disabled"
    for profile_name in ("quick", "full"):
        profile = data["profiles"][profile_name]
        assert "steps" in profile
        for step in profile["steps"]:
            assert step.get("id")
            assert step.get("name")
            assert step.get("command")
            assert "outputs" in step
            assert step["verify"]["type"] in ("exit_code", "gate_script")


def test_manifest_gate_floors_documented():
    data = json.loads(MANIFEST.read_text())
    phi_steps = []
    for profile in data["profiles"].values():
        for step in profile["steps"]:
            gates = step.get("gates", {})
            if "phi_iq_floor" in gates:
                phi_steps.append(step)
                assert gates["phi_iq_floor"] == "logs/benchmark_ci_baseline.json"
                assert gates["tolerance"] == 0.05
    assert len(phi_steps) >= 2
    for step in data["profiles"]["full"]["steps"]:
        if step["id"] == "nightly_stress":
            assert step["gates"]["leak_slope_late_b_per_cyc"] == 1600
            assert step["gates"]["leak_slope_fill_b_per_cyc"] == 5000


def test_dry_run_quick_profile():
    proc = subprocess.run(
        [sys.executable, str(REPRODUCE_SCRIPT), "--dry-run", "--profile", "quick"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "lint" in proc.stdout
    assert "benchmark_quick_gate" in proc.stdout
    assert "anomaly_gate" in proc.stdout
    data = json.loads(MANIFEST.read_text())
    assert len(data["profiles"]["quick"]["steps"]) >= 7


def test_dry_run_full_profile():
    proc = subprocess.run(
        [sys.executable, str(REPRODUCE_SCRIPT), "--dry-run", "--profile", "full"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "benchmark_canonical" in proc.stdout
    assert "causal_gate" in proc.stdout
    assert "nightly_stress" in proc.stdout
    data = json.loads(MANIFEST.read_text())
    assert len(data["profiles"]["full"]["steps"]) >= 11


def test_verify_gate_script_helpers():
    mod = _load_reproduce_module()
    baseline = {
        "overall_phi_iq": 0.6,
        "config": {},
        "results": [],
    }
    report = {
        "overall_phi_iq": 0.58,
        "config": {},
        "results": [],
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        baseline_p = tmp_path / "baseline.json"
        report_p = tmp_path / "report.json"
        baseline_p.write_text(json.dumps(baseline))
        report_p.write_text(json.dumps(report))

        step = {
            "verify": {
                "type": "gate_script",
                "argv": [
                    sys.executable,
                    str(ROOT / "scripts" / "check_benchmark_gate.py"),
                    str(report_p),
                    str(baseline_p),
                ],
            },
        }
        env = mod.build_env({"environment": {"defaults": {}}}, {})
        ok, rc, _ = mod.verify_step(step, 0, env, ROOT)
        assert ok is True
        assert rc == 0

        bad_report = tmp_path / "bad.json"
        bad_report.write_text(json.dumps({"overall_phi_iq": 0.4}))
        step_fail = {
            "verify": {
                "type": "gate_script",
                "argv": [
                    sys.executable,
                    str(ROOT / "scripts" / "check_benchmark_gate.py"),
                    str(bad_report),
                    str(baseline_p),
                ],
            },
        }
        ok_fail, rc_fail, _ = mod.verify_step(step_fail, 0, env, ROOT)
        assert ok_fail is False
        assert rc_fail == 1


def test_report_writer_structure():
    mod = _load_reproduce_module()
    manifest = mod.load_manifest(MANIFEST)
    result = mod.StepResult(
        id="lint",
        name="Python lint",
        status="PASS",
        started_at="2026-07-04T00:00:00+00:00",
        duration_s=1.5,
        exit_code=0,
        verify_pass=True,
        command="ruff check python/",
        outputs=[],
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "report.json"
        mod.write_report(
            manifest, "quick", [result], "PASS",
            {"python": "3.11.0", "mujoco_gl": "disabled"},
            out,
        )
        report = json.loads(out.read_text())
        assert report["overall"] == "PASS"
        assert report["profile"] == "quick"
        assert len(report["steps"]) == 1
        assert report["steps"][0]["id"] == "lint"
