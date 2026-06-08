"""The GUI is a thin stdlib-http layer over run_leaderboard_job. Its testable cores — the run
registry (background thread + log capture + stop), roster loading, API-key detection, and result-path
safety — are unit-tested without sockets; one socket smoke test confirms the server answers.
"""
from __future__ import annotations

import threading
import time

from inversa import gui


def test_run_registry_captures_log_and_result():
    reg = gui.RunRegistry()

    def fake_runner(params, on_log=None, abort=None):
        on_log("started")
        on_log("done")
        return {"results": [{"model": "a", "igs": 1.0}], "meta": {}}

    rid = reg.start({"models": ["a"]}, runner=fake_runner)
    for _ in range(200):
        if reg.state(rid)["status"] in ("done", "error"):
            break
        time.sleep(0.01)
    st = reg.state(rid)
    assert st["status"] == "done"
    assert st["log"] == ["started", "done"]
    assert st["result"]["results"][0]["model"] == "a"


def test_run_registry_stop_signals_abort():
    reg = gui.RunRegistry()
    saw = {}

    def waiting_runner(params, on_log=None, abort=None):
        saw["abort"] = abort
        while not abort.is_set():
            time.sleep(0.005)
        on_log("aborted")
        return {"results": [], "meta": {}}

    rid = reg.start({"models": ["a"]}, runner=waiting_runner)
    time.sleep(0.05)
    reg.stop(rid)
    for _ in range(200):
        if reg.state(rid)["status"] == "done":
            break
        time.sleep(0.01)
    assert saw["abort"].is_set()
    assert reg.state(rid)["status"] == "done"


def test_run_registry_records_error():
    reg = gui.RunRegistry()

    def boom(params, on_log=None, abort=None):
        raise RuntimeError("kaboom")

    rid = reg.start({"models": ["a"]}, runner=boom)
    for _ in range(200):
        if reg.state(rid)["status"] == "error":
            break
        time.sleep(0.01)
    assert reg.state(rid)["status"] == "error"
    assert "kaboom" in reg.state(rid)["error"]


def test_is_local_request_allows_localhost_and_blocks_csrf_and_rebinding():
    f = gui.is_local_request
    # same-origin / no Origin from 127.0.0.1 or localhost -> allowed
    assert f("127.0.0.1:8000", None) is True
    assert f("localhost:8000", "http://127.0.0.1:8000") is True
    assert f("127.0.0.1:8000", "http://localhost:8000") is True
    # DNS-rebinding: a foreign Host that resolves to 127.0.0.1 -> blocked
    assert f("evil.example.com:8000", None) is False
    # CSRF: a real site POSTing cross-origin carries a foreign Origin -> blocked
    assert f("127.0.0.1:8000", "https://evil.example.com") is False


def test_load_roster_is_a_list():
    r = gui.load_roster()
    assert isinstance(r, list)


def test_safe_result_path_blocks_traversal():
    assert gui.safe_result_path("../../etc/passwd") is None
    assert gui.safe_result_path("/abs/escape") is None
    p = gui.safe_result_path("igs_leaderboard_30.json")
    assert p is not None and "results" in str(p).replace("\\", "/")


def test_server_smoke(tmp_path):
    server, port = gui.make_server(port=0)  # port 0 -> ephemeral
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        import json
        import urllib.request
        base = f"http://127.0.0.1:{port}"
        home = urllib.request.urlopen(base + "/", timeout=5).read().decode("utf-8")
        assert "Inversa" in home
        cfg = json.loads(urllib.request.urlopen(base + "/config", timeout=5).read())
        assert "key_present" in cfg
        roster = json.loads(urllib.request.urlopen(base + "/roster", timeout=5).read())
        assert isinstance(roster, list)
    finally:
        server.shutdown()
