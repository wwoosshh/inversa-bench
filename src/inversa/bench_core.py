"""Reusable, network-free cores for the Inversa leaderboard engine. The CLI orchestration
(cli_bench) stays a thin wrapper so the expensive parts are exercised by fast unit tests.

Why these exist — a 100-model x (30 pose + 30 transform) x N-repeat leaderboard is dominated
by wall-clock and API spend, so the engine needs:
  A. global item-level concurrency: schedule every (model, item) call in ONE bounded pool so a
     slow reasoning model can't hold a worker hostage for a whole model's 60 sequential calls.
  E. discrimination pruning: items every model passes (or every model fails) cost a call but
     carry zero ranking information — drop them after a first pass.
  C. adaptive repeats: spend extra repeats only on models whose rank is statistically
     ambiguous, instead of paying 3x across the whole roster.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def map_concurrent(fn: Callable[[T], R], items: Sequence[T],
                   max_workers: int) -> List[Optional[R]]:
    """Apply `fn` to every item across a pool of at most `max_workers` threads, returning
    results in INPUT order. A failing item resolves to None instead of aborting the batch —
    one hung/erroring model call must not lose the other results in a long leaderboard run.
    """
    if not items:
        return []

    results: List[Optional[R]] = [None] * len(items)

    def _run(idx_item):
        idx, item = idx_item
        try:
            results[idx] = fn(item)
        except Exception:
            results[idx] = None

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(items)))) as ex:
        list(ex.map(_run, enumerate(items)))
    return results


def discriminating_item_indices(matrix: Sequence[Sequence[bool]]) -> List[int]:
    """Given a model x item validity matrix, return the column indices that DISCRIMINATE —
    columns where models disagree (at least one pass and one fail). A column every model
    passes, or every model fails, costs a call but adds zero ranking information, so dropping
    it shrinks every subsequent pass. Returns [] for an empty matrix or a single model (no
    disagreement is observable with one row).
    """
    if not matrix:
        return []
    n_cols = len(matrix[0])
    kept: List[int] = []
    for c in range(n_cols):
        col = [bool(row[c]) for row in matrix]
        if any(col) and not all(col):
            kept.append(c)
    return kept


def _stderr(score: float, n: int) -> float:
    """Binomial standard error of a validity rate measured over `n` machine-verified items.
    Zero at the 0/1 boundaries — a model that passes everything (or nothing) is decided."""
    if n <= 0:
        return float("inf")
    p = min(1.0, max(0.0, score))
    return (p * (1.0 - p) / n) ** 0.5


def ambiguous_models(scored: Sequence[Tuple[str, float, int]], z: float = 1.96) -> List[str]:
    """Pick the models whose leaderboard RANK is not yet statistically resolved, so repeats are
    spent only where they change the ordering. Two adjacent-ranked models are indistinguishable
    when their score gap is within z*(se_i + se_j) (default z=1.96 -> ~95%). A model is flagged
    if it ties this way with either neighbor. Well-separated models — and settled 0/1 scores
    (se=0) — are never flagged, so the bulk of a tier-spread roster is done in one pass.
    """
    ranked = sorted(scored, key=lambda s: -s[1])
    flagged = set()
    for i in range(len(ranked) - 1):
        (mi, pi, ni), (mj, pj, nj) = ranked[i], ranked[i + 1]
        if abs(pi - pj) <= z * (_stderr(pi, ni) + _stderr(pj, nj)):
            flagged.add(mi)
            flagged.add(mj)
    return [m for m, _p, _n in ranked if m in flagged]


def item_signature(item: Dict[str, Any]) -> str:
    """Content hash of one bank item, independent of key order. Folds the item's content into
    the cache key so editing a bank item naturally invalidates its stale cached result."""
    canonical = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def cache_key(model: str, bank: str, rep: int, item: Dict[str, Any]) -> str:
    """Identify one machine-verified call uniquely by (model, bank, repeat index, item content).
    Including `rep` keeps repeated samples distinct, so resume never collapses N repeats into
    one cached answer (which would silently defeat the noise reduction repeats buy)."""
    return f"{model}\x1f{bank}\x1f{rep}\x1f{item_signature(item)}"


class ResultCache:
    """Item-level result store so a re-run — after a crash, a deadline abandon, an added model,
    or an added repeat — skips the calls it already paid for. A flat dict keyed by cache_key(),
    persisted as JSON. Value is whatever the engine stores per item (e.g. {"valid": bool})."""

    def __init__(self, data: Optional[Dict[str, Any]] = None, path: Optional[str] = None) -> None:
        self._data: Dict[str, Any] = dict(data or {})
        self._path = path
        # the engine writes the cache from many worker threads while another thread may persist
        # it mid-run; the lock keeps put/save from racing (e.g. dict-changed-during-iteration).
        self._lock = threading.Lock()

    @classmethod
    def load(cls, path: str) -> "ResultCache":
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return cls(json.load(f), path=path)
        return cls({}, path=path)

    def get(self, key: str) -> Optional[Any]:
        return self._data.get(key)

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def save(self, path: Optional[str] = None) -> None:
        target = path or self._path
        if not target:
            raise ValueError("ResultCache.save needs a path (none given at load or save time)")
        with self._lock:
            snapshot = dict(self._data)  # snapshot under lock; do file IO unlocked
        tmp = f"{target}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False)
        os.replace(tmp, target)  # atomic: a crash mid-write can't corrupt the resume file


def discrimination_matrix(cache: "ResultCache", models: Sequence[str], bank: str,
                          items: Sequence[Dict[str, Any]], rep: int = 0) -> List[List[bool]]:
    """Rebuild the model x item validity matrix for one bank from a resume cache (repeat `rep`),
    so a finished run can be mined for which items actually discriminate (-> discriminating_item_
    indices) without paying for the calls again. Models missing any item are dropped (an
    incomplete row would make an item look falsely dead)."""
    rows: List[List[bool]] = []
    for m in models:
        hits = [cache.get(cache_key(m, bank, rep, it)) for it in items]
        if any(h is None for h in hits):
            continue
        rows.append([bool(h["valid"]) for h in hits])
    return rows


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


_FATAL_MARKERS = ("402", "401", "insufficient credit", "insufficient_quota",
                  "authentication", "invalid api key", "no auth credentials")


def is_fatal_account_error(exc: BaseException) -> bool:
    """True for account-WIDE failures (out of credit, bad/zeroed key) where every further call
    will fail identically — so the engine should abort rather than grind thousands of retried
    failures (which is exactly how a mid-run credit exhaustion silently burned a real run's
    repeats). A per-model 404/timeout is NOT fatal: other models can still be measured."""
    msg = str(exc).lower()
    return any(marker in msg for marker in _FATAL_MARKERS)


def evaluate_leaderboard(models: Sequence[str], pose_items: Sequence[Dict[str, Any]],
                         transform_items: Sequence[Dict[str, Any]],
                         score_item: Callable[[str, str, Dict[str, Any], int], bool],
                         *, repeats: int = 1, max_workers: int = 8,
                         cache: Optional[ResultCache] = None, adaptive: bool = True,
                         on_rep_done: Optional[Callable[[int], None]] = None,
                         log: Optional[Callable[[str], None]] = None) -> List[Dict[str, Any]]:
    """Score a roster on the two IGS banks with the three efficiency levers wired together:

      A (concurrency): every (model, bank, item) call for a repeat is scheduled in ONE bounded
        pool, so fast models free their slots immediately and a slow model only holds its own.
      D (resume): each call is memoized in `cache` keyed by (model, bank, rep, item); a re-run
        after a crash/abandon/added-model pays only for the calls it hasn't made yet.
      C (adaptive repeats): repeat 0 runs everyone; further repeats (up to `repeats`) are spent
        only on models whose rank is still statistically ambiguous, not the whole roster.

    `score_item(model, bank, item, rep) -> bool` does the actual generate+verify (injected so the
    engine is network-free and unit-tested). It MAY raise on infra failure; a model whose every
    call fails is skipped (not recorded as a real zero). Returns per-model dicts sorted by IGS.
    """
    from inversa.scoring import igs as _igs

    _log = log or (lambda _m: None)
    banks = (("pose", pose_items), ("transform", transform_items))
    pose_rates: Dict[str, List[float]] = {m: [] for m in models}
    trans_rates: Dict[str, List[float]] = {m: [] for m in models}
    reps_done: Dict[str, int] = {m: 0 for m in models}
    broken: set = set()
    abort = threading.Event()  # tripped by an account-wide fatal error (out of credit / bad key)

    def _run_rep(rep: int, rep_models: Sequence[str]):
        """Returns (attempted, ok) call counts for this repeat so the caller can see failures."""
        units = [(m, bank, it) for m in rep_models for bank, items in banks for it in items]

        def work(unit):
            m, bank, it = unit
            if abort.is_set():
                return (m, bank, False, False)  # account-wide failure already seen; don't call
            key = cache_key(m, bank, rep, it) if cache is not None else None
            if cache is not None:
                hit = cache.get(key)
                if hit is not None:
                    if hit.get("missing"):
                        return (m, bank, False, False)  # cached as not-answered -> excluded
                    return (m, bank, True, bool(hit["valid"]))
            try:
                v = score_item(m, bank, it, rep)  # None => item not answered (truncated/missing)
            except Exception as e:
                if is_fatal_account_error(e):
                    abort.set()  # every further call will fail the same way — stop scheduling
                return (m, bank, False, False)  # ok=False: infra failure, not a wrong answer
            if v is None:  # missing: excluded from the rate denominator, not scored as wrong (#10)
                if cache is not None:
                    cache.put(key, {"missing": True})
                return (m, bank, False, False)
            valid = bool(v)
            if cache is not None:
                cache.put(key, {"valid": valid})
            return (m, bank, True, valid)

        # tallies[model][bank] = [n_valid, n_ok]
        tallies = {m: {"pose": [0, 0], "transform": [0, 0]} for m in rep_models}
        attempted = ok_total = 0
        for r in map_concurrent(work, units, max_workers):
            if r is None:
                continue
            m, bank, ok, valid = r
            attempted += 1
            if ok:
                ok_total += 1
                tallies[m][bank][1] += 1
                tallies[m][bank][0] += int(valid)

        for m in rep_models:
            p_valid, p_ok = tallies[m]["pose"]
            t_valid, t_ok = tallies[m]["transform"]
            if p_ok == 0 and t_ok == 0:
                broken.add(m)
                continue
            pose_rates[m].append(p_valid / p_ok if p_ok else 0.0)
            trans_rates[m].append(t_valid / t_ok if t_ok else 0.0)
            reps_done[m] += 1
        return attempted, ok_total

    att0, ok0 = _run_rep(0, list(models))
    if on_rep_done:
        on_rep_done(0)
    if att0 and ok0 < att0:
        _log(f"[warn] rep 0: {att0 - ok0}/{att0} calls failed "
             f"({len([m for m in models if m in broken])} models unmeasurable)")
    if abort.is_set():
        _log("[abort] account-wide fatal error (out of credit / bad key) during rep 0 — "
             "results are PARTIAL; add credit and re-run with the same --cache to finish.")

    alive = [m for m in models if m not in broken]
    n_items = len(pose_items) + len(transform_items)
    for rep in range(1, max(1, repeats)):
        if abort.is_set():
            break
        if adaptive:
            scored = [(m, _igs(_mean(pose_rates[m]), _mean(trans_rates[m])),
                       n_items * reps_done[m]) for m in alive]
            wanted = set(ambiguous_models(scored))
            rep_models = [m for m in alive if m in wanted]
        else:
            rep_models = alive
        if not rep_models:
            break
        att, ok = _run_rep(rep, rep_models)
        if on_rep_done:
            on_rep_done(rep)
        if att and ok == 0:
            _log(f"[abort] repeat {rep}: all {att} calls failed — stopping extra repeats "
                 "(likely out of credit / rate-limited). Earlier repeats are kept.")
            break

    out: List[Dict[str, Any]] = []
    for m in alive:
        if not pose_rates[m]:
            continue
        pv, tv = _mean(pose_rates[m]), _mean(trans_rates[m])
        out.append({"model": m, "pose_validity": round(pv, 4),
                    "transform_validity": round(tv, 4), "igs": _igs(pv, tv),
                    "reps_done": reps_done[m], "n_pose": len(pose_items),
                    "n_transform": len(transform_items)})
    out.sort(key=lambda r: -r["igs"])
    return out
