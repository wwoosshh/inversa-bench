"""Reusable benchmark-engine cores that make a 100-model x 60-item leaderboard affordable:
global item-level concurrency (A), item-discrimination pruning (E), adaptive repeats (C),
and an item-level result cache for resume (D). Each is a pure, network-free unit so the
expensive CLI orchestration stays a thin wrapper over tested logic.
"""
from __future__ import annotations

import threading
import time

from inversa.bench_core import (
    ResultCache,
    ambiguous_models,
    cache_key,
    discrimination_matrix,
    discriminating_item_indices,
    evaluate_leaderboard,
    is_fatal_account_error,
    item_signature,
    map_concurrent,
)


def test_map_concurrent_preserves_input_order():
    # completion order != input order, but results must line up with inputs
    def fn(x):
        time.sleep(0.01 * (3 - x))  # later items finish first
        return x * x

    assert map_concurrent(fn, [1, 2, 3], max_workers=3) == [1, 4, 9]


def test_map_concurrent_bounds_simultaneous_workers():
    active = 0
    peak = 0
    lock = threading.Lock()

    def fn(_x):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return _x

    map_concurrent(fn, list(range(20)), max_workers=4)
    assert peak <= 4


def test_map_concurrent_isolates_failures_as_none_without_aborting_the_batch():
    # one bad item (a hung/erroring model call) must not lose the other 99 results
    def fn(x):
        if x == 2:
            raise ValueError("boom")
        return x

    assert map_concurrent(fn, [1, 2, 3], max_workers=2) == [1, None, 3]


def test_map_concurrent_empty_input():
    assert map_concurrent(lambda x: x, [], max_workers=4) == []


# --- E: item-discrimination pruning ----------------------------------------

def test_discriminating_indices_keeps_only_columns_with_disagreement():
    # cols: 0 = all pass (dead), 1 = all fail (dead), 2 = mixed (discriminates)
    matrix = [
        [True, False, True],
        [True, False, False],
        [True, False, True],
    ]
    assert discriminating_item_indices(matrix) == [2]


def test_discriminating_indices_all_dead_returns_empty():
    matrix = [[True, False], [True, False]]
    assert discriminating_item_indices(matrix) == []


def test_discriminating_indices_keeps_all_when_every_column_splits():
    matrix = [[True, True], [False, False]]
    assert discriminating_item_indices(matrix) == [0, 1]


def test_discriminating_indices_empty_matrix():
    assert discriminating_item_indices([]) == []


# --- C: adaptive repeats (only re-run statistically ambiguous ranks) --------

def _names(scored):
    return sorted(m for m in scored)


def test_well_separated_models_need_no_extra_repeats():
    # 0.10 vs 0.90 over 60 items: CIs nowhere near overlapping
    scored = [("a", 0.90, 60), ("b", 0.10, 60)]
    assert ambiguous_models(scored) == []


def test_near_tied_models_are_flagged_for_more_repeats():
    # 0.50 vs 0.52 over 60 items: indistinguishable -> both need more data
    scored = [("a", 0.52, 60), ("b", 0.50, 60)]
    assert _names(ambiguous_models(scored)) == ["a", "b"]


def test_only_the_close_adjacent_pair_is_flagged():
    # c is close to b (0.50/0.48) but a (0.95) is clear of both
    scored = [("a", 0.95, 60), ("b", 0.50, 60), ("c", 0.48, 60)]
    assert _names(ambiguous_models(scored)) == ["b", "c"]


def test_settled_models_at_zero_and_one_are_never_flagged():
    # se=0 at the boundaries -> a perfect and a zero model are decided in one pass
    scored = [("a", 1.0, 30), ("b", 0.0, 30)]
    assert ambiguous_models(scored) == []


def test_single_model_has_no_neighbor_to_be_ambiguous_with():
    assert ambiguous_models([("a", 0.5, 60)]) == []


# --- D: item-level result cache for resume ----------------------------------

def test_item_signature_is_stable_and_key_order_independent():
    a = item_signature({"target_desc": "sqrt(2)+1", "novelty": "irr", "v": 2.41})
    b = item_signature({"v": 2.41, "novelty": "irr", "target_desc": "sqrt(2)+1"})
    assert a == b


def test_item_signature_distinguishes_different_items():
    assert item_signature({"source": "x**3-2"}) != item_signature({"source": "x**3-3"})


def test_cache_key_separates_model_bank_rep_and_item():
    item = {"source": "x**3-2"}
    base = cache_key("gpt", "transform", 0, item)
    assert base != cache_key("claude", "transform", 0, item)   # model
    assert base != cache_key("gpt", "pose", 0, item)           # bank
    assert base != cache_key("gpt", "transform", 1, item)      # rep -> repeats stay distinct


def test_cache_roundtrip_and_miss():
    c = ResultCache({})
    c.put("k1", {"valid": True})
    assert c.get("k1") == {"valid": True}
    assert c.get("missing") is None


def test_cache_persists_and_resumes_from_disk(tmp_path):
    path = tmp_path / "cache.json"
    c = ResultCache.load(str(path))
    c.put("k1", {"valid": False})
    c.save()

    resumed = ResultCache.load(str(path))
    assert resumed.get("k1") == {"valid": False}


def test_cache_load_missing_file_starts_empty(tmp_path):
    c = ResultCache.load(str(tmp_path / "does_not_exist.json"))
    assert c.get("anything") is None


# --- engine: evaluate_leaderboard (A + C + D wired together) -----------------

def _items(n):
    return [{"i": k} for k in range(n)]


def _rate_scorer(rates):
    """score_item that makes each model hit an exact validity rate per bank, deterministically."""
    def score_item(model, bank, item, rep):
        p = rates[model]
        return (item["i"] % 50) < round(p * 50)  # exact rate for 50- or 100-item banks
    return score_item


def test_engine_aggregates_igs_per_model():
    pose, trans = _items(100), _items(100)
    res = evaluate_leaderboard(["a", "b"], pose, trans,
                               _rate_scorer({"a": 0.90, "b": 0.30}),
                               repeats=1, max_workers=8, adaptive=False)
    by = {r["model"]: r for r in res}
    assert by["a"]["pose_validity"] == 0.90
    assert by["a"]["igs"] == 0.90
    assert by["b"]["igs"] == 0.30
    assert res[0]["model"] == "a"  # sorted by igs desc


def test_engine_uses_cache_and_skips_already_scored_items():
    pose, trans = _items(10), _items(10)
    cache = ResultCache({})
    for bank, items in (("pose", pose), ("transform", trans)):
        for it in items:
            cache.put(cache_key("a", bank, 0, it), {"valid": True})

    def forbidden(model, bank, item, rep):
        raise AssertionError("score_item must not be called on a full cache hit")

    res = evaluate_leaderboard(["a"], pose, trans, forbidden,
                               repeats=1, max_workers=4, cache=cache, adaptive=False)
    assert res[0]["igs"] == 1.0


def test_engine_writes_results_into_cache_for_resume():
    pose, trans = _items(10), _items(10)
    cache = ResultCache({})
    evaluate_leaderboard(["a"], pose, trans, _rate_scorer({"a": 1.0}),
                         repeats=1, max_workers=4, cache=cache, adaptive=False)
    # a second run hitting a raising scorer must succeed entirely from cache
    res = evaluate_leaderboard(["a"], pose, trans,
                               lambda *a: (_ for _ in ()).throw(AssertionError("no calls")),
                               repeats=1, max_workers=4, cache=cache, adaptive=False)
    assert res[0]["igs"] == 1.0


def test_engine_spends_extra_repeats_only_on_ambiguous_ranks():
    pose, trans = _items(50), _items(50)
    res = evaluate_leaderboard(["a", "b", "c"], pose, trans,
                               _rate_scorer({"a": 0.90, "b": 0.52, "c": 0.50}),
                               repeats=3, max_workers=8, adaptive=True)
    reps = {r["model"]: r["reps_done"] for r in res}
    assert reps["a"] == 1          # clearly on top -> decided in one pass
    assert reps["b"] == 3 and reps["c"] == 3  # near-tied -> spend the repeat budget


def test_engine_skips_models_whose_calls_all_fail():
    pose, trans = _items(20), _items(20)

    def scorer(model, bank, item, rep):
        if model == "bad":
            raise RuntimeError("model unavailable (404)")
        return True

    res = evaluate_leaderboard(["good", "bad"], pose, trans, scorer,
                               repeats=1, max_workers=8, adaptive=False)
    models = {r["model"] for r in res}
    assert models == {"good"}  # a model that errors on every call is skipped, not scored 0


# --- account-wide fatal errors: surface + fast-abort (credit exhaustion) -----

def test_is_fatal_account_error_flags_credit_and_auth_only():
    assert is_fatal_account_error(RuntimeError("Error code: 402 - Insufficient credits"))
    assert is_fatal_account_error(RuntimeError("401 invalid api key"))
    # per-model / transient failures must NOT trip a global abort
    assert not is_fatal_account_error(RuntimeError("Error code: 404 - model not found"))
    assert not is_fatal_account_error(RuntimeError("Connection timed out"))


def test_engine_fast_aborts_after_account_wide_credit_error():
    # a 402 means every further call will 402 too; the engine must stop scheduling, not grind
    # thousands of retried failures (what silently burned the real run's repeats).
    calls = {"n": 0}
    lock = threading.Lock()

    def scorer(model, bank, item, rep):
        with lock:
            calls["n"] += 1
        raise RuntimeError("Error code: 402 - Insufficient credits")

    pose, trans = _items(50), _items(50)  # 3 models x 100 items = 300 units
    res = evaluate_leaderboard(["a", "b", "c"], pose, trans, scorer,
                               repeats=1, max_workers=4, adaptive=False)
    assert res == []                 # every model failed -> nothing measured
    assert calls["n"] < 100          # aborted early; nowhere near all 300 units attempted


# --- E wiring: rebuild the model x item matrix from the resume cache ----------

def _fill(cache, model, bank, items, verdicts):
    for it, v in zip(items, verdicts):
        cache.put(cache_key(model, bank, 0, it), {"valid": v})


def test_discrimination_matrix_reconstructs_rows_from_cache():
    cache = ResultCache({})
    items = [{"i": 0}, {"i": 1}, {"i": 2}]
    _fill(cache, "a", "pose", items, [True, False, True])
    _fill(cache, "b", "pose", items, [True, False, False])

    matrix = discrimination_matrix(cache, ["a", "b"], "pose", items)
    assert matrix == [[True, False, True], [True, False, False]]
    assert discriminating_item_indices(matrix) == [2]  # only item 2 separates a from b


def test_discrimination_matrix_skips_models_with_incomplete_cache():
    cache = ResultCache({})
    items = [{"i": 0}, {"i": 1}]
    _fill(cache, "a", "pose", items, [True, True])
    _fill(cache, "b", "pose", items[:1], [True])  # b missing item 1 -> excluded

    assert discrimination_matrix(cache, ["a", "b"], "pose", items) == [[True, True]]


def test_result_cache_counts_hits_and_misses():
    # the GUI reports cache hits vs fresh calls; ResultCache must tally get() outcomes
    c = ResultCache({"a": {"valid": True}})
    assert c.get("a") == {"valid": True}   # hit
    assert c.get("nope") is None           # miss
    assert c.get("a") is not None          # hit
    assert (c.hits, c.misses) == (2, 1)
