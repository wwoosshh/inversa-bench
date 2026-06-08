"""The benchmark run logic is shared by the CLI and the GUI via run_leaderboard_job, so it must be
callable in-process with an injected adapter factory (no network), stream progress through on_log,
write the result files, and honor an external abort (the GUI's Stop button).
"""
from __future__ import annotations

import json
import threading

from inversa.adapters.fake import FakeAdapter
from inversa.bench_run import run_leaderboard_job


def _banks(tmp_path):
    pose = tmp_path / "pose.json"
    pose.write_text(json.dumps({"targets": [{"target_desc": "1", "target_value": 1.0,
                                             "novelty": "integer"}]}), encoding="utf-8")
    trans = tmp_path / "trans.json"
    trans.write_text(json.dumps({"items": [{"source": "x**3 + x - 1 = 0", "g_desc": "r + 1",
                                            "r_value": 0.6823, "target_value": 1.6823}]}),
                     encoding="utf-8")
    return str(pose), str(trans)


def _params(tmp_path, **over):
    pose, trans = _banks(tmp_path)
    p = {"models": ["a", "b"], "pose_bank": pose, "transform_bank": trans,
         "repeats": 1, "adaptive": False, "cache": "",
         "out": str(tmp_path / "o.html"), "json_out": str(tmp_path / "o.json")}
    p.update(over)
    return p


def test_job_runs_end_to_end_with_injected_adapter(tmp_path):
    logs = []
    out = run_leaderboard_job(_params(tmp_path), on_log=logs.append,
                              adapter_factory=lambda m: FakeAdapter(["#### x - 1 = 0"]))
    assert {r["model"] for r in out["results"]} == {"a", "b"}
    # target 1 -> "x - 1 = 0" is a valid unique pose; transform target 1.68 -> invalid
    by = {r["model"]: r for r in out["results"]}
    assert by["a"]["pose_validity"] == 1.0
    assert (tmp_path / "o.json").exists() and (tmp_path / "o.html").exists()
    assert logs  # progress / summary streamed, not printed


def test_job_writes_meta_and_ranked_json(tmp_path):
    run_leaderboard_job(_params(tmp_path), adapter_factory=lambda m: FakeAdapter(["#### x - 1 = 0"]))
    data = json.load(open(tmp_path / "o.json", encoding="utf-8"))
    assert data["n_models"] == 2 and "results" in data
    assert data["results"][0]["rank"] == 1


def test_external_abort_stops_the_run(tmp_path):
    ev = threading.Event()
    ev.set()  # already aborted -> every item short-circuits, nothing measured
    out = run_leaderboard_job(_params(tmp_path), abort=ev,
                              adapter_factory=lambda m: FakeAdapter(["#### x - 1 = 0"]))
    assert out["results"] == []


def test_job_records_difficulty_label_in_meta(tmp_path):
    out = run_leaderboard_job(_params(tmp_path, difficulty="brutal"),
                              adapter_factory=lambda m: FakeAdapter(["#### x - 1 = 0"]))
    assert out["meta"]["difficulty"] == "brutal"


def test_job_reports_cache_stats_on_resume(tmp_path):
    p = _params(tmp_path, cache=str(tmp_path / "c.json"))
    fac = lambda m: FakeAdapter(["#### x - 1 = 0"])  # noqa: E731
    run_leaderboard_job(p, adapter_factory=fac)            # first run: all misses -> live calls
    out = run_leaderboard_job(p, adapter_factory=fac)      # second run: all cache hits
    assert out["cache_stats"]["hits"] > 0
