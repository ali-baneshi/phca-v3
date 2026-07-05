"""Unit tests for session_io JSONL loaders."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from phca.monitoring.session_io import (
    frame_from_json,
    frame_to_json,
    load_jsonl_lines,
    load_jsonl_path,
    load_session_frames,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "reacher_short.jsonl"


def test_frame_from_json_roundtrip_scalar_and_array():
    raw = {
        "schema_version": 1,
        "cycle_id": 3,
        "env_kind": "mujoco_rgb",
        "prediction_error": 1.5,
        "obs_vector": [1.0, 0.0, 0.5],
        "drive_targets": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    }
    f = frame_from_json(raw)
    assert f.cycle_id == 3
    assert f.prediction_error == 1.5
    assert f.obs_vector is not None
    assert len(f.drive_targets) == 6
    out = frame_to_json(f)
    assert out["cycle_id"] == 3
    assert out["drive_targets"] == raw["drive_targets"]


def test_load_jsonl_lines_skips_blank():
    lines = ["", "  ", json.dumps({"cycle_id": 0, "schema_version": 1})]
    frames = load_jsonl_lines(lines)
    assert len(frames) == 1
    assert frames[0].cycle_id == 0


def test_load_jsonl_path_and_session_dir(tmp_path):
    src = _FIXTURE.read_text()
    jsonl = tmp_path / "timeseries.jsonl"
    jsonl.write_text(src)
    frames_path = load_jsonl_path(jsonl)
    sess = tmp_path
    frames_sess = load_session_frames(sess)
    assert len(frames_path) == len(frames_sess)
    assert frames_path[0].env_kind == "mujoco_rgb"


def test_frame_from_json_deep_copies_nested():
    raw = {
        "cycle_id": 0,
        "schema_version": 1,
        "action_rationale": {"best_score": 0.4, "goal_id": 2},
        "candidate_scores": [0.4, 0.2],
    }
    f = frame_from_json(raw)
    f.action_rationale["best_score"] = 9.9
    f.candidate_scores[0] = 9.9
    assert raw["action_rationale"]["best_score"] == 0.4
    assert raw["candidate_scores"][0] == 0.4


def test_frame_from_json_grid_array():
    grid = [[0, 1], [2, 3]]
    f = frame_from_json({"cycle_id": 0, "schema_version": 1, "grid": grid})
    assert isinstance(f.grid, np.ndarray)
    assert f.grid.dtype == np.int32
    assert int(f.grid[1, 0]) == 2
