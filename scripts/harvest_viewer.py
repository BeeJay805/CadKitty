"""Live harvest viewer + CONTROL PANEL for the 4090 flywheel harvest.

Serves http://127.0.0.1:8765 : shows the newest HIGH-FIDELITY part the local model produced
(shipped AND the fidelity gate did not flag a primitive pile), tracks live numbers (build / ship /
FIDELITY pass-fail-pile, mean score, per-tier, dataset rows, ETA), and OWNS the harvest process so
you can START / PAUSE it from the page. The server being the single owner is also what prevents the
zombie-loop stacking we hit before: one server, one harvest child, killed as a tree on pause.

Read-only on the data; the only side effect is spawning/killing the harvest subprocess. Run:
    python scripts/harvest_viewer.py [port]
"""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "benchmarks" / "harvest" / "work"
LOG = REPO / "benchmarks" / "harvest" / "harvest_log.jsonl"
BANK = REPO / "benchmarks" / "harvest" / "prompts_complex.jsonl"  # TEACHER runs the complex bank
PY = REPO / ".venv" / "Scripts" / "python.exe"
_MODE = "teacher"  # full-Claude production pipeline distilling high-fidelity, non-primitive targets

# production-faithful flags (mirror fly.toml). TEACHER authors on the Claude subscription (Opus) and
# writes its distilled dataset to a SEPARATE file so it stays clean of the qwen self-harvest.
_HARVEST_ENV = {
    "UV_PROJECT_ENVIRONMENT": str(REPO / ".venv"),
    "FORGE_LLM_PROVIDER": "claude_code",
    "FORGE_FORBID_API_SPEND": "1",
    "FORGE_TRAIN_LOG": "1",
    "FORGE_TRAIN_LOG_PATH": str(Path.home() / ".forge-train" / "teacher.jsonl"),
    "FORGE_HARVEST_BANK": str(BANK),
    "OLLAMA_NUM_PARALLEL": "1",
    "OLLAMA_MAX_LOADED_MODELS": "1",
    "FORGE_PARAM_OPT": "1",
    "FORGE_PARAM_OPT_EVALS": "24",
    "FORGE_PARAM_OPT_WORKERS": "2",
    "FORGE_FEATURE_PLAN": "1",
    "FORGE_AUTHOR_MAX_TOKENS": "16000",
    "FORGE_PHYSICS_ROUTE": "1",
    "PYTHONPATH": str(REPO / "scripts"),
}


# --- harvest process controller (server is the single owner) ------------------------------------
_lock = threading.Lock()
_running = False
_proc: subprocess.Popen | None = None
_thread: threading.Thread | None = None


def _all_done() -> bool:
    total = _bank_count()
    return total > 0 and _real_rows() >= total


def _harvest_loop() -> None:
    global _proc
    env = {**os.environ, **_HARVEST_ENV}
    while _running:
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        proc = subprocess.Popen(
            [str(PY), "-u", "scripts/harvest.py", _MODE, "1"],
            cwd=str(REPO),
            env=env,
            creationflags=flags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with _lock:
            _proc = proc
        proc.wait()
        with _lock:
            _proc = None
        if not _running or proc.returncode == 0 or _all_done():
            break  # clean completion of all prompts (rc 0) or paused; else a crash -> respawn
        time.sleep(3)


def _kill_proc() -> None:
    with _lock:
        p = _proc
    if p is None:
        return
    with contextlib.suppress(Exception):
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True, check=False
            )
        else:
            p.terminate()


def start_harvest() -> str:
    global _running, _thread
    with _lock:
        if _running:
            return "already running"
        _running = True
    _thread = threading.Thread(target=_harvest_loop, daemon=True)
    _thread.start()
    return "started"


def pause_harvest() -> str:
    global _running
    with _lock:
        was = _running
        _running = False
    _kill_proc()
    return "paused" if was else "was not running"


def is_running() -> bool:
    with _lock:
        return _running and _proc is not None and _proc.poll() is None


atexit.register(pause_harvest)


# --- data readers -------------------------------------------------------------------------------
def _rows() -> list[dict]:
    out: list[dict] = []
    try:
        for line in LOG.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    except OSError:
        pass
    return out


def _real_rows() -> int:
    return sum(1 for r in _rows() if not r.get("infra_fail") and r.get("planner") == _MODE)


def _bank_count() -> int:
    try:
        return sum(1 for x in BANK.read_text(encoding="utf-8").splitlines() if x.strip())
    except OSError:
        return 0


def _dataset_count() -> int:
    try:
        return sum(
            1 for _ in (Path.home() / ".forge-train" / "teacher.jsonl").open(encoding="utf-8")
        )
    except OSError:
        return 0


def _prompts() -> dict[str, str]:
    d: dict[str, str] = {}
    try:
        for line in BANK.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                d[r["id"]] = r["prompt"]
    except OSError:
        pass
    return d


def _good_dirs(rows: list[dict]) -> set[str]:
    """Work-dir names that are HIGH FIDELITY: shipped AND the fidelity gate did not fail/flag a
    primitive pile. Only these reach the screen -- drafts, no-builds, rejects, and primitive piles
    are hidden."""
    ok: set[str] = set()
    for r in rows:
        if r.get("infra_fail") or r.get("planner") != _MODE:
            continue
        shipped = r.get("disposition") in ("shipped", "shipped_with_caveats")
        clean = r.get("fid_verdict") not in ("fail",) and not r.get("fid_pile")
        if shipped and clean:
            ok.add(str(r.get("key", "")).replace("|", "_"))
    return ok


def _newest_png(rows: list[dict]) -> Path | None:
    if not WORK.exists():
        return None
    good = _good_dirs(rows)
    best: tuple[float, Path] | None = None
    for p in WORK.glob("*/part.png"):
        if p.parent.name not in good:
            continue
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if best is None or m > best[0]:
            best = (m, p)
    return best[1] if best else None


def _state() -> dict:
    rows = _rows()
    real = [r for r in rows if not r.get("infra_fail") and r.get("planner") == _MODE]
    png = _newest_png(rows)
    by_row = {str(r.get("key", "")).replace("|", "_"): r for r in rows}
    cur = {}
    if png is not None:
        key = png.parent.name
        m = by_row.get(key, {})
        pid = "_".join(key.split("_")[1:-1])
        cur = {
            "family": m.get("family", key),
            "tier": m.get("tier", "?"),
            "disposition": m.get("disposition"),
            "fid_score": m.get("fid_score"),
            "fid_expected": m.get("fid_expected"),
            "seconds": m.get("seconds"),
            "prompt": _prompts().get(pid, ""),
            "dir": key,
            "mtime": png.stat().st_mtime,
        }
    per: dict[int, list[int]] = {}
    for r in real:
        t = r.get("tier", 0)
        p, n = per.get(t, [0, 0])
        per[t] = [p + (1 if r.get("hard_ok") else 0), n + 1]
    built = sum(1 for r in real if r.get("hard_ok"))
    shipped = sum(1 for r in real if r.get("disposition") in ("shipped", "shipped_with_caveats"))
    fid_pass = sum(1 for r in real if r.get("fid_verdict") == "pass")
    fid_fail = sum(1 for r in real if r.get("fid_verdict") == "fail")
    fid_pile = sum(1 for r in real if r.get("fid_pile"))
    high_fid = len(_good_dirs(rows))
    scored = [r["fid_score"] for r in real if isinstance(r.get("fid_score"), (int, float))]
    mean_fid = round(sum(scored) / len(scored), 2) if scored else 0.0
    secs = [r["seconds"] for r in real if r.get("seconds")]
    mean_sec = sum(secs) / len(secs) if secs else 0.0
    total_prompts = _bank_count()
    remaining = max(0, total_prompts - len(real))
    last_build = 0.0
    with contextlib.suppress(OSError):
        for pp in WORK.glob("*/part.png"):
            last_build = max(last_build, pp.stat().st_mtime)
    return {
        "running": is_running(),
        "has_png": png is not None,
        "cur": cur,
        "totals": {t: per[t] for t in sorted(per)},
        "n": len(real),
        "built": built,
        "shipped": shipped,
        "high_fid": high_fid,
        "fid_pass": fid_pass,
        "fid_fail": fid_fail,
        "fid_pile": fid_pile,
        "mean_fid": mean_fid,
        "dataset": _dataset_count(),
        "total_prompts": total_prompts,
        "remaining": remaining,
        "mean_sec": round(mean_sec, 1),
        "eta_sec": round(remaining * mean_sec),
        "server_now": time.time(),
        "last_build_mtime": last_build,
    }


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>Forge-Mind - harvest control</title><style>
:root{color-scheme:dark light}
body{margin:0;font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;background:#0b0e13;color:#d7dde5;
 display:flex;flex-direction:column;align-items:center;min-height:100vh}
header{width:100%;text-align:center;padding:12px 0 6px;border-bottom:1px solid #1e2733}
h1{margin:0;font-size:15px;letter-spacing:.4px;color:#8ab4f8;font-weight:600}
#sub{margin-top:3px;font-size:11px;color:#f0a35e;letter-spacing:.3px;text-transform:uppercase}
#ctl{display:flex;gap:12px;align-items:center;margin:14px 0 6px}
button{font:600 13px ui-monospace,monospace;padding:9px 22px;border-radius:8px;border:1px solid #2b3a4d;
 background:#141b25;color:#d7dde5;cursor:pointer}
button:hover{background:#1b2431}#run{border-color:#1f6f43}#run.on{background:#0f2e1b;color:#4ade80}
#pause{border-color:#6f2020}#status{font-size:12px;color:#7d8899}
#img{max-width:min(84vw,720px);max-height:52vh;margin:12px auto 6px;border:1px solid #263140;
 border-radius:10px;background:#11151c;box-shadow:0 8px 30px #0008;display:none}
#waiting{margin:40px auto;color:#7d8899;text-align:center}
#meta{max-width:720px;width:84vw;background:#11161e;border:1px solid #1e2733;border-radius:10px;padding:10px 15px;margin:4px auto}
.row{display:flex;justify-content:space-between;gap:12px;padding:2px 0}.k{color:#7d8899}.v{color:#e6edf3;text-align:right}
.pill{padding:2px 9px;border-radius:20px;font-weight:600;font-size:12px}
.ship{background:#0f2e1b;color:#4ade80;border:1px solid #1f6f43}
#prompt{color:#9aa7b6;font-size:13px;margin-top:6px;font-style:italic}
#cards{max-width:860px;width:92vw;display:flex;gap:9px;justify-content:center;margin:8px 0 18px;flex-wrap:wrap}
.card{background:#11161e;border:1px solid #1e2733;border-radius:8px;padding:8px 13px;text-align:center;min-width:70px}
.card b{color:#8ab4f8;font-size:16px}.card span{color:#7d8899;font-size:10px;display:block;text-transform:uppercase;letter-spacing:.3px}
.bad b{color:#f87171}.good b{color:#4ade80}
#stamp{color:#586170;font-size:11px;margin-bottom:14px}
@media(prefers-color-scheme:light){body{background:#f6f8fa;color:#1f2733}header,#meta,.card{border-color:#e1e6eb;background:#fff}
 #img{background:#fff;border-color:#d6dde5}.k,.card span{color:#6b7684}.v{color:#111}}
</style></head><body>
<header><h1>FORGE-MIND - TEACHER harvest (Claude Opus author + seeing critic/judge -> complex parts)</h1>
<div id=sub>HIGH-FIDELITY ONLY: shipped &amp; passed the anti-primitive gate &mdash; drafts / piles / rejects hidden</div></header>
<div id=ctl><button id=run onclick=go('start')>&#9654; START</button>
<button id=pause onclick=go('pause')>&#10073;&#10073; PAUSE</button><span id=status>-</span></div>
<img id=img alt="latest high-fidelity part">
<div id=waiting>no high-fidelity part yet &mdash; press START to harvest</div>
<div id=meta>
 <div class=row><span class=k>part</span><span class=v id=fam>-</span></div>
 <div class=row><span class=k>verdict</span><span class=v id=verdict>-</span></div>
 <div class=row><span class=k>fidelity</span><span class=v id=fid>-</span></div>
 <div class=row><span class=k>last build</span><span class=v id=lastbuild>-</span></div>
 <div id=prompt>-</div>
</div>
<div id=cards></div>
<div id=stamp>-</div>
<script>
function ago(s){if(!s)return '-';s=Math.max(0,Math.floor(s));if(s<60)return s+'s ago';
 if(s<3600)return Math.floor(s/60)+'m ago';return Math.floor(s/3600)+'h '+Math.floor((s%3600)/60)+'m ago';}
function dur(s){if(s==null)return '-';s=Math.round(s);if(s<3600)return Math.floor(s/60)+'m';
 return Math.floor(s/3600)+'h '+Math.floor((s%3600)/60)+'m';}
async function go(cmd){await fetch('/'+cmd,{method:'POST'});tick();}
async function tick(){
 try{
  const s=await (await fetch('/state.json?t='+Date.now())).json();
  const run=document.getElementById('run');
  run.classList.toggle('on',s.running);
  document.getElementById('status').textContent=s.running?'harvesting - one LLM at a time':'paused';
  document.getElementById('img').style.display=s.has_png?'block':'none';
  document.getElementById('waiting').style.display=s.has_png?'none':'block';
  if(s.has_png){const c=s.cur;
   document.getElementById('img').src='/latest.png?t='+Math.floor(c.mtime);
   document.getElementById('fam').textContent=(c.family||'-')+'  (tier '+c.tier+', exp '+c.fid_expected+')';
   document.getElementById('verdict').innerHTML='<span class="pill ship">SHIPPED - real construction</span>';
   document.getElementById('fid').textContent='score '+(c.fid_score!=null?c.fid_score:'-');
   document.getElementById('lastbuild').textContent=s.last_build_mtime?
     (new Date(s.last_build_mtime*1000).toLocaleTimeString()+'  ('+ago(s.server_now-s.last_build_mtime)+')'):'-';
   document.getElementById('prompt').textContent='"'+(c.prompt||c.dir)+'"';
  }
  const cards=[
   ['high-fidelity', s.high_fid, 'good'],
   ['built', s.built+'/'+s.n, ''],
   ['shipped', s.shipped+'/'+s.n, ''],
   ['fidelity PASS', s.fid_pass, 'good'],
   ['primitive FAIL', s.fid_fail+s.fid_pile, 'bad'],
   ['mean score', s.mean_fid, ''],
   ['dataset rows', s.dataset, ''],
   ['done / total', s.n+' / '+s.total_prompts, ''],
   ['est. time left', dur(s.eta_sec), ''],
  ];
  const box=document.getElementById('cards');box.innerHTML='';
  for(const [lab,val,cls] of cards){const d=document.createElement('div');
   d.className='card'+(cls?' '+cls:'');d.innerHTML='<b>'+val+'</b><span>'+lab+'</span>';box.appendChild(d);}
  const tl=[];for(const[t,pr]of Object.entries(s.totals))tl.push('T'+t+' '+pr[0]+'/'+pr[1]);
  document.getElementById('stamp').textContent='per-tier build: '+(tl.join('  ')||'-')+
   '   |   updated '+new Date().toLocaleTimeString()+' - auto 3s';
 }catch(e){document.getElementById('status').textContent='server unreachable';}
}
tick();setInterval(tick,3000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        with contextlib.suppress(Exception):
            self.wfile.write(body)

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/start":
            msg = start_harvest()
        elif path == "/pause":
            msg = pause_harvest()
        else:
            self._send(404, b"no", "text/plain")
            return
        self._send(200, json.dumps({"ok": True, "msg": msg}).encode(), "application/json")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/state.json":
            self._send(200, json.dumps(_state()).encode("utf-8"), "application/json")
        elif path == "/latest.png":
            png = _newest_png(_rows())
            if png is None:
                self._send(404, b"", "text/plain")
                return
            try:
                self._send(200, png.read_bytes(), "image/png")
            except OSError:
                self._send(404, b"", "text/plain")
        else:
            self._send(404, b"", "text/plain")


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"harvest control on http://127.0.0.1:{port}  (Ctrl-C to stop)", flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        srv.serve_forever()
    pause_harvest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
