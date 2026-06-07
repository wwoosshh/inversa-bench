"""Inversa local web GUI — zero new dependencies (Python stdlib `http.server` only).

`python -m inversa.gui` starts a localhost-only server and opens the browser. The page drives the
SAME engine as the CLI (`bench_run.run_leaderboard_job`) in a background thread, streaming progress
and showing the ranked leaderboard — an easy on-ramp for researchers who don't want to memorize
flags, with an Advanced panel exposing every option.

Testable cores (RunRegistry, load_roster, key_present, safe_result_path) are unit-tested without
sockets; the HTTP handler is a thin router over them.
"""
from __future__ import annotations

import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

from inversa.bench_run import run_leaderboard_job

_RESULTS_DIR = Path("data/results")
_ROSTER = Path("data/banks/leaderboard_roster.json")


def key_present() -> bool:
    """True if an OpenRouter key is available (from .env/env) — so the UI can hide the key field."""
    load_dotenv()
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def load_roster() -> list:
    """The built-in 102-model roster for the model picker (empty list if the file is absent)."""
    try:
        data = json.load(open(_ROSTER, encoding="utf-8"))
        return data if isinstance(data, list) else data.get("models", [])
    except Exception:
        return []


def safe_result_path(rel: str) -> Optional[Path]:
    """Resolve `rel` strictly inside data/results (None on traversal/absolute escape) so the
    /file endpoint cannot serve arbitrary files."""
    if not rel:
        return None
    base = _RESULTS_DIR.resolve()
    target = (base / rel).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        return None
    return target


def list_results() -> list:
    """Existing leaderboard/dashboard artifacts a user can open from the GUI."""
    if not _RESULTS_DIR.exists():
        return []
    out = []
    for f in sorted(_RESULTS_DIR.glob("*.html")):
        out.append({"name": f.name, "path": f.name})
    return out


class RunRegistry:
    """Tracks background benchmark runs: log capture, status, result, and a per-run abort Event
    (the Stop button). `runner` is injectable so tests run with no network."""

    def __init__(self) -> None:
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._n = 0

    def start(self, params: Dict[str, Any],
              runner: Callable[..., Dict[str, Any]] = run_leaderboard_job) -> str:
        with self._lock:
            self._n += 1
            rid = str(self._n)
            st = {"status": "running", "log": [], "result": None, "error": None,
                  "abort": threading.Event()}
            self._runs[rid] = st

        def _work():
            try:
                res = runner(params, on_log=lambda m: st["log"].append(m), abort=st["abort"])
                st["result"] = res
                st["status"] = "done"
            except Exception as e:  # noqa: BLE001
                st["error"] = f"{type(e).__name__}: {e}"
                st["status"] = "error"

        threading.Thread(target=_work, daemon=True).start()
        return rid

    def state(self, rid: str) -> Dict[str, Any]:
        return self._runs.get(rid, {"status": "unknown", "log": [], "result": None, "error": None})

    def stop(self, rid: str) -> bool:
        st = self._runs.get(rid)
        if st and st.get("abort"):
            st["abort"].set()
            return True
        return False


_REGISTRY = RunRegistry()

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Inversa — Generative Score</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,500;1,9..144,600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --ink:#080b14; --panel:rgba(255,255,255,.035); --panel2:rgba(255,255,255,.06);
  --line:rgba(120,180,255,.12); --grid:rgba(120,170,255,.045);
  --text:#e8ecf6; --muted:#8089a0; --accent:#5eead4; --accent-dim:rgba(94,234,212,.16);
  --amber:#f5b342; --rose:#fb7185;
  --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
  --disp:'Fraunces',Georgia,'Times New Roman',serif;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;min-height:100vh;color:var(--text);font-family:var(--mono);font-size:14px;line-height:1.55;
  background:
    radial-gradient(1100px 540px at 78% -12%, rgba(94,234,212,.10), transparent 62%),
    radial-gradient(900px 500px at 8% 8%, rgba(120,170,255,.07), transparent 60%),
    linear-gradient(var(--grid) 1px, transparent 1px) 0 0/30px 30px,
    linear-gradient(90deg, var(--grid) 1px, transparent 1px) 0 0/30px 30px,
    var(--ink);
  background-attachment:fixed;}
.wrap{max-width:860px;margin:0 auto;padding:2.4rem 1.2rem 4rem}
::selection{background:var(--accent);color:#04221d}

/* header */
header{margin-bottom:1.8rem;opacity:0;animation:rise .7s .02s cubic-bezier(.2,.7,.2,1) forwards}
.mark{font-family:var(--disp);font-weight:600;font-size:clamp(44px,9vw,86px);line-height:.92;
  letter-spacing:-.02em;font-optical-sizing:auto}
.mark em{font-style:italic;color:var(--accent);position:relative}
.mark .swap{display:inline-block;color:var(--muted);font-size:.42em;font-style:normal;font-family:var(--mono);
  transform:translateY(-.55em);margin-left:.1em}
.rule{height:1px;background:linear-gradient(90deg,var(--accent),transparent 65%);margin:.7rem 0 1rem}
.tag{color:var(--muted);font-size:13.5px;max-width:60ch}
.tag b{color:var(--text);font-weight:600}
.badges{margin-top:.7rem;display:flex;gap:.5rem;flex-wrap:wrap}
.badge{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);
  border:1px solid var(--line);border-radius:999px;padding:3px 10px;background:var(--accent-dim)}

/* cards */
.card{position:relative;border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin:1rem 0;
  background:var(--panel);backdrop-filter:blur(6px);
  opacity:0;animation:rise .7s cubic-bezier(.2,.7,.2,1) forwards}
.card:nth-of-type(1){animation-delay:.08s}.card:nth-of-type(2){animation-delay:.16s}
.card:nth-of-type(3){animation-delay:.24s}.card:nth-of-type(4){animation-delay:.30s}
.step{font-family:var(--disp);font-style:italic;color:var(--accent);font-size:18px;margin-right:.35rem}
@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}

.banner{border-radius:10px;padding:10px 13px;font-size:13.5px;margin:.2rem 0 0;border:1px solid var(--line)}
.ok{background:var(--accent-dim);border-color:rgba(94,234,212,.4);color:#bff7ec}
.warn{background:rgba(245,179,66,.1);border-color:rgba(245,179,66,.4);color:#fbd38d}

label{font-size:12px;color:var(--muted);display:block;margin:.7rem 0 .25rem;letter-spacing:.02em}
input[type=text],input[type=password],input[type=number],textarea,select{width:100%;padding:9px 11px;
  border:1px solid var(--line);border-radius:8px;font:inherit;font-size:13.5px;color:var(--text);
  background:rgba(0,0,0,.25);transition:border-color .15s,box-shadow .15s}
input:focus,textarea:focus,select:focus{outline:0;border-color:var(--accent);
  box-shadow:0 0 0 3px var(--accent-dim)}
select option{background:#0d1320}
textarea{resize:vertical}
.row{display:flex;gap:14px;flex-wrap:wrap}.row>div{flex:1;min-width:130px}
label.chk{display:flex;align-items:center;gap:.5rem;color:var(--text);font-size:13px;margin:.55rem 0}
label.chk input{width:auto}

button{font:inherit;font-weight:600;font-size:14px;border:0;border-radius:9px;padding:11px 22px;cursor:pointer;
  background:var(--accent);color:#04221d;letter-spacing:.01em;transition:transform .12s,box-shadow .2s,filter .2s}
button:hover{transform:translateY(-1px);box-shadow:0 8px 26px -8px var(--accent),0 0 0 1px var(--accent)}
button:disabled{filter:grayscale(.6) brightness(.7);cursor:default;transform:none;box-shadow:none}
button.sec{background:transparent;color:var(--muted);border:1px solid var(--line);padding:6px 13px;
  font-weight:500;border-radius:7px}
button.sec:hover{color:var(--accent);border-color:var(--accent);box-shadow:none;transform:none}

#models{max-height:236px;overflow:auto;border:1px solid var(--line);border-radius:9px;padding:8px 10px;
  background:rgba(0,0,0,.22);margin-top:.4rem}
#models::-webkit-scrollbar{width:9px}#models::-webkit-scrollbar-thumb{background:var(--line);border-radius:9px}
.fam{font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);opacity:.75;
  margin:.7rem 0 .2rem;font-weight:600}.fam:first-child{margin-top:.1rem}
.m{font-size:13px;display:flex;align-items:center;gap:.5rem;padding:2px 0;color:#cfd6e6;cursor:pointer}
.m:hover{color:#fff} .m input{width:auto;accent-color:var(--accent)}
.selcount{color:var(--accent);font-size:12px;margin-left:.6rem}

details summary{cursor:pointer;color:var(--muted);font-size:12.5px;letter-spacing:.1em;text-transform:uppercase;
  list-style:none;user-select:none;margin:.3rem 0}
details summary::-webkit-details-marker{display:none}
details[open] summary{color:var(--accent)}

pre{background:linear-gradient(180deg,#05080f,#070b15);color:#9ff5e2;border:1px solid var(--line);
  border-radius:11px;padding:14px 15px;overflow:auto;max-height:320px;font-size:12px;line-height:1.65;
  white-space:pre-wrap;word-break:break-word}
pre::-webkit-scrollbar{width:9px}pre::-webkit-scrollbar-thumb{background:rgba(94,234,212,.25);border-radius:9px}

table{border-collapse:collapse;width:100%;margin:.6rem 0}
th,td{border-bottom:1px solid var(--line);padding:7px 9px;font-size:13px;text-align:left}
th{font-size:10.5px;letter-spacing:.1em;color:var(--muted);text-transform:uppercase;font-weight:500}
td:first-child{color:var(--accent);font-weight:600} .c{text-align:center}
tbody tr:hover{background:var(--panel2)}
a{color:var(--accent);text-decoration:none;border-bottom:1px solid var(--accent-dim)}
a:hover{border-color:var(--accent)}
.muted{color:var(--muted);font-size:12px}
#runcard.live{border-color:rgba(94,234,212,.45);box-shadow:0 0 0 1px var(--accent-dim),0 0 40px -16px var(--accent)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--accent);margin-right:6px;
  animation:pulse 1.1s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:.35;transform:scale(.8)}50%{opacity:1;transform:scale(1.15)}}
</style></head><body><div class="wrap">

<header>
  <div class="mark">Invers<em>a</em> <span class="swap">answer&#8594;problem</span></div>
  <div class="rule"></div>
  <p class="tag">A contamination-resistant, <b>machine-verified</b> math benchmark. Give a model an
  answer; it must <b>construct</b> a problem that yields it — and a sympy oracle checks it, with
  <b>no LLM judge</b>. Score = <b>IGS = mean(pose, transform)</b>.</p>
  <div class="badges"><span class="badge">sympy-verified</span><span class="badge">no LLM judge</span>
  <span class="badge">contamination-immune</span></div>
</header>

<div id="keybanner"></div>

<div class="card">
  <label><span class="step">01</span>Pick models <span class="muted">(or add your own below)</span></label>
  <input type="text" id="filter" placeholder="filter…  gpt · claude · qwen · deepseek" oninput="renderModels()">
  <div style="margin:.5rem 0"><button class="sec" onclick="pick('all')">select all</button>
    <button class="sec" onclick="pick('none')">clear</button>
    <span id="selcount" class="selcount"></span></div>
  <div id="models"></div>
  <label>Extra models — comma-separated, any OpenRouter id</label>
  <textarea id="extra" rows="2" placeholder="openai/gpt-4o-mini, anthropic/claude-haiku-4.5, …"></textarea>

  <div id="keyfield" style="display:none"><label><span class="step">02</span>OpenRouter API key
    <span class="muted">— used only for this run</span></label>
    <input type="password" id="apikey" placeholder="sk-or-…"></div>

  <details><summary>▸ Advanced options</summary>
    <div class="row">
      <div><label>repeats (max)</label><input type="number" id="repeats" value="1" min="1"></div>
      <div><label>max workers</label><input type="number" id="workers" value="12" min="1"></div>
      <div><label>max tokens</label><input type="number" id="maxtok" value="8000"></div>
      <div><label>reasoning cap (0=off)</label><input type="number" id="rcap" value="4000"></div>
    </div>
    <div class="row">
      <div><label>temperature</label><input type="number" id="temp" value="0" step="0.1"></div>
      <div><label>pose mode</label><select id="posemode">
        <option value="bank30">fixed bank · 30 items</option>
        <option value="bank6">fixed bank · 6 items</option>
        <option value="random">random N · contamination-immune</option></select></div>
      <div><label>pose random N</label><input type="number" id="poseN" value="30"></div>
    </div>
    <label class="chk"><input type="checkbox" id="adaptive" checked> adaptive repeats (only ambiguous ranks)</label>
    <label class="chk"><input type="checkbox" id="notrivial"> reject trivial pose (anti-gaming)</label>
    <label class="chk"><input type="checkbox" id="truncmiss"> treat truncation as missing (don't deflate reasoning models)</label>
  </details>

  <div style="margin-top:1.1rem"><button id="runbtn" onclick="run()">Run benchmark &#8594;</button>
    <span id="costnote" class="muted"></span></div>
</div>

<div class="card" id="runcard" style="display:none">
  <b><span class="step" style="margin:0">▸</span> Run <span id="rstatus"></span></b>
  <button class="sec" id="stopbtn" onclick="stop()" style="float:right">Stop</button>
  <div id="resultbox"></div>
  <pre id="log"></pre>
</div>

<div class="card"><details><summary>▸ Past results</summary><div id="pastresults" style="margin-top:.5rem"></div></details></div>
</div>

<script>
let ROSTER=[], SEL=new Set(), RID=null, POLL=null;
async function init(){
  const cfg=await (await fetch('/config')).json();
  document.getElementById('keybanner').innerHTML = cfg.key_present
    ? '<div class="banner ok">✓ API key detected from .env — ready to run.</div>'
    : '<div class="banner warn">No API key found. Enter your OpenRouter key below (used only for this run).</div>';
  if(!cfg.key_present) document.getElementById('keyfield').style.display='block';
  ROSTER=await (await fetch('/roster')).json();
  renderModels(); loadPast();
}
function renderModels(){
  const f=document.getElementById('filter').value.toLowerCase();
  const box=document.getElementById('models'); box.innerHTML='';
  let fam='';
  ROSTER.filter(m=>m.toLowerCase().includes(f)).forEach(m=>{
    const fm=m.split('/')[0];
    if(fm!==fam){fam=fm; const h=document.createElement('div'); h.className='fam'; h.textContent=fm; box.appendChild(h);}
    const id='cb_'+m;
    const l=document.createElement('label'); l.className='m';
    l.innerHTML='<input type="checkbox" '+(SEL.has(m)?'checked':'')+' onchange="toggle(\\''+m+'\\',this.checked)"> '+m;
    box.appendChild(l);
  });
  document.getElementById('selcount').textContent=SEL.size+' selected';
}
function toggle(m,on){on?SEL.add(m):SEL.delete(m); document.getElementById('selcount').textContent=SEL.size+' selected';}
function pick(w){ROSTER.filter(m=>m.toLowerCase().includes(document.getElementById('filter').value.toLowerCase()))
  .forEach(m=>{w==='all'?SEL.add(m):SEL.delete(m);}); renderModels();}
function models(){
  const extra=document.getElementById('extra').value.split(',').map(s=>s.trim()).filter(Boolean);
  return Array.from(new Set([...SEL, ...extra]));
}
async function run(){
  const ms=models();
  if(!ms.length){alert('Pick at least one model.');return;}
  if(!confirm(ms.length+' models × ~60 items — this makes real, paid API calls. Continue?'))return;
  const pm=document.getElementById('posemode').value;
  const params={models:ms,
    api_key:document.getElementById('apikey')?document.getElementById('apikey').value||null:null,
    repeats:+document.getElementById('repeats').value, adaptive:document.getElementById('adaptive').checked,
    max_workers:+document.getElementById('workers').value, max_tokens:+document.getElementById('maxtok').value,
    reasoning_max_tokens:+document.getElementById('rcap').value, temperature:+document.getElementById('temp').value,
    pose_no_trivial:document.getElementById('notrivial').checked, truncation_missing:document.getElementById('truncmiss').checked,
    pose_bank: pm==='bank6'?'data/banks/struct_pose_targets.json':'data/banks/struct_pose_targets30.json',
    pose_random: pm==='random'?+document.getElementById('poseN').value:0,
    out:'data/results/igs_gui.html', json_out:'data/results/igs_gui.json'};
  document.getElementById('runbtn').disabled=true;
  const r=await (await fetch('/run',{method:'POST',body:JSON.stringify(params)})).json();
  RID=r.run_id; const rc=document.getElementById('runcard'); rc.style.display='block'; rc.classList.add('live');
  document.getElementById('stopbtn').style.display='';
  document.getElementById('resultbox').innerHTML='';
  POLL=setInterval(poll,1200); poll();
}
async function poll(){
  if(!RID)return;
  const s=await (await fetch('/progress?id='+RID)).json();
  document.getElementById('rstatus').innerHTML = s.status==='running'
    ? '<span class="dot"></span>running' : '— '+s.status;
  document.getElementById('log').textContent=s.log.join('\\n');
  document.getElementById('log').scrollTop=1e9;
  if(s.status==='done'||s.status==='error'){
    clearInterval(POLL); document.getElementById('runbtn').disabled=false;
    document.getElementById('runcard').classList.remove('live');
    document.getElementById('stopbtn').style.display='none';
    if(s.status==='error'){document.getElementById('resultbox').innerHTML='<div class="banner warn">Error: '+s.error+'</div>';}
    else renderResult(s.result); loadPast();
  }
}
function renderResult(res){
  const rows=(res.results||[]).map(r=>'<tr><td>'+r.rank+'</td><td>'+r.model+'</td><td class=c>'+r.igs.toFixed(2)
    +'</td><td class=c>'+r.pose_validity.toFixed(2)+'</td><td class=c>'+r.transform_validity.toFixed(2)
    +'</td><td class=c>'+r.reps_done+'</td></tr>').join('');
  document.getElementById('resultbox').innerHTML=
    '<p><a href="/file?path=igs_gui.html" target="_blank">open leaderboard ↗</a> · '
    +'<a href="/file?path=igs_gui.json" target="_blank">download JSON ↗</a></p>'
    +'<table><tr><th>#</th><th>model</th><th>IGS</th><th>pose</th><th>transform</th><th>reps</th></tr>'+rows+'</table>';
}
async function stop(){if(RID)await fetch('/stop?id='+RID,{method:'POST'});}
async function loadPast(){
  const r=await (await fetch('/results')).json();
  document.getElementById('pastresults').innerHTML=r.length
    ? r.map(f=>'<div><a href="/file?path='+encodeURIComponent(f.path)+'" target="_blank">'+f.name+'</a></div>').join('')
    : '<span class="muted">none yet</span>';
}
init();
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet console
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if u.path == "/config":
            return self._send(200, json.dumps({"key_present": key_present()}))
        if u.path == "/roster":
            return self._send(200, json.dumps(load_roster()))
        if u.path == "/results":
            return self._send(200, json.dumps(list_results()))
        if u.path == "/progress":
            rid = parse_qs(u.query).get("id", [""])[0]
            st = _REGISTRY.state(rid)
            return self._send(200, json.dumps({"status": st["status"], "log": st["log"],
                                               "result": st["result"], "error": st["error"]}))
        if u.path == "/file":
            rel = parse_qs(u.query).get("path", [""])[0]
            p = safe_result_path(rel)
            if p is None or not p.exists():
                return self._send(404, json.dumps({"error": "not found"}))
            ctype = "text/html; charset=utf-8" if p.suffix == ".html" else "application/json"
            return self._send(200, p.read_bytes(), ctype)
        return self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/run":
            n = int(self.headers.get("Content-Length", 0))
            params = json.loads(self.rfile.read(n) or b"{}")
            if not params.get("models"):
                return self._send(400, json.dumps({"error": "no models selected"}))
            rid = _REGISTRY.start(params)
            return self._send(200, json.dumps({"run_id": rid}))
        if u.path == "/stop":
            rid = parse_qs(u.query).get("id", [""])[0]
            return self._send(200, json.dumps({"stopped": _REGISTRY.stop(rid)}))
        return self._send(404, json.dumps({"error": "not found"}))


def make_server(port: int = 8000):
    """Build a localhost-only threading HTTP server; port 0 picks a free port. Returns (server, port)."""
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    return server, server.server_address[1]


def main(argv=None) -> None:
    load_dotenv()
    port = 8000
    server = None
    for p in range(8000, 8010):  # find a free port if the default is busy
        try:
            server, port = make_server(p)
            break
        except OSError:
            continue
    if server is None:
        server, port = make_server(0)
    url = f"http://127.0.0.1:{port}/"
    print(f"Inversa GUI → {url}  (Ctrl+C to stop)", flush=True)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
