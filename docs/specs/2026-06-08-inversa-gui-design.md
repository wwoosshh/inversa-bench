# Inversa local web GUI — design spec

**Goal:** make Inversa runnable by a non-CLI researcher in minutes from a fresh clone. The CLI's
multi-flag command is fine for tooling but a barrier for people; a GUI that fully controls the engine
(easy mode by default, advanced panel for every flag) is the accessible entry point.

## Decisions (from brainstorming)
- **Zero-dependency local web GUI** — Python stdlib `http.server` only; no new pip deps.
- **Easy mode + advanced panel** — minimal inputs by default; all flags available when expanded.

## Architecture
- `python -m inversa.gui` → starts `http.server` on `127.0.0.1:<port>` (default 8000; auto-increment if
  busy), opens the browser. **localhost only**, never bound to 0.0.0.0.
- Single self-contained page (HTML/CSS/JS embedded in Python; no external assets/CDN).
- Endpoints:
  - `GET /` — control page.
  - `GET /roster` — built-in 102-model roster (from `data/banks/leaderboard_roster.json`) as JSON.
  - `GET /config` — whether `OPENROUTER_API_KEY` is present (via `.env`/env); never returns the key.
  - `POST /run` — JSON body of run params → starts a background job, returns `{run_id}`.
  - `GET /progress?id=` — `{status, log[], partial_results[], usage, done}` for polling (~1s).
  - `POST /stop?id=` — request abort of a running job.
  - `GET /results` — list existing dashboards/leaderboards under `data/results/`.
  - `GET /file?path=` — serve a result file (path-validated to `data/results/`).

## Shared engine job (refactor)
Extract the run logic currently inline in `cli_bench.main` into
`inversa/bench_run.py::run_leaderboard_job(params, on_log=None, abort=None, adapter_factory=None)`:
- Builds adapters (or uses an injected `adapter_factory` for tests/no-network), defines `score_item`,
  loads/builds banks (incl. `--pose-random`), runs `evaluate_leaderboard`, writes JSON+HTML+dashboard,
  returns the ranked results + usage. `on_log(str)` streams progress; `abort` is a `threading.Event`.
- `cli_bench.main` becomes a thin wrapper that parses args → `params` → calls the job (no duplicated
  scoring/adapter code). GUI calls the same job in a thread.
- `params` is a plain dict mirroring the CLI flags (models, key, banks, pose_random, pose_seed,
  pose_no_trivial, repeats, adaptive, max_workers, max_tokens, reasoning_max_tokens, temperature,
  truncation_missing, cache, out, json_out).

## UI
- **Easy mode (default):** model picker (checkbox list from /roster + free-text add) · API-key field
  (shows "detected from .env" if present, else input) · **Run**.
- **Advanced (collapsed):** all remaining flags with sensible defaults and short help text.
- **Running:** live log pane (from /progress), progress count, partial ranked table, token spend, **Stop**.
- **Done:** final IGS table + "Open dashboard" / "Download JSON". A "Past results" tab lists existing
  dashboards via /results.
- **Cost confirm:** before POST /run, a modal shows "N models × (pose+transform items) ≈ ~K calls —
  this spends real API credit. Proceed?".

## Key & safety
- Key resolution order: per-run field (if provided) → `.env`/env. Per-run key used in-memory only; a
  "save to .env" checkbox is the ONLY way it's persisted.
- Server bound to 127.0.0.1; `/file` validates the path stays within `data/results/` (no traversal).
- Missing key / out-of-credit / port-in-use → explicit friendly messages.

## Onboarding
- README top: **"Quick start (GUI)"** — `pip install -e .` → `python -m inversa.gui` → browser. CLI
  moves below as the power-user path.

## Testing
- `run_leaderboard_job` with an injected FakeAdapter factory → end-to-end, no network; assert results,
  on_log called, outputs written, `abort` stops it.
- GUI: factor request handling so config-parse / roster / key-detection are unit-testable without
  sockets; one socket smoke test (`GET /` → 200 + form markers, `/roster` → JSON, `/config` → status).

## Scope (YAGNI)
- **In:** everything above (easy+advanced, live progress, stop, results browser, onboarding).
- **Out (later):** dry-run cost estimation, persistent multi-run history, auth/remote exposure.
