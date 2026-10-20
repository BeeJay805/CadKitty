#!/usr/bin/env python3
"""Forge-Mind Operator Console -- a local, live status dashboard.

    python scripts/operator_console.py          # then open http://localhost:8787
    python scripts/operator_console.py 9000     # custom port

Shows, refreshed live (so you don't have to ask in chat): open PRs + their CI / merge state,
recent merges to main, recent CI runs, open issues (health alerts + filed feedback), the North
Star ladder (parsed from BACKLOG.md), the pipeline + guard state, and whether the working tree is
clean. Reads git + the `gh` CLI + the repo docs on every request; binds localhost only. Stdlib
only -- no extra dependencies, so it runs anywhere the repo + gh are set up.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _run(args: list[str], timeout: int = 25) -> str:
    try:
        out = subprocess.run(args, cwd=REPO, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _gh(args: list[str]) -> list[dict]:
    raw = _run(["gh", *args])
    try:
        data = json.loads(raw) if raw else []
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _parse_backlog() -> dict[str, list[dict]]:
    try:
        txt = (REPO / "docs" / "forge-mind" / "BACKLOG.md").read_text(encoding="utf-8")
    except Exception:
        return {}
    buckets: dict[str, list[dict]] = {}
    current: str | None = None
    for line in txt.splitlines():
        header = re.match(r"^#{2,3}\s+(.*)", line)
        if header:
            current = header.group(1).strip()
            buckets[current] = []
            continue
        item = re.match(r"^\s*-\s*\[([ x])\]\s*\*\*(.+?)\*\*", line)
        if item and current is not None:
            buckets[current].append({"done": item.group(1) == "x", "title": item.group(2).strip()})

    def find(key: str) -> list[dict]:
        for name, rows in buckets.items():
            if key.lower() in name.lower():
                return rows
        return []

    return {
        "feedback": find("FOUNDER FEEDBACK"),
        "now": find("Now"),
        "next": find("Next"),
        "later": find("Later"),
        "done": find("Done"),
    }


def _pipeline_state() -> dict:
    guard = (Path.home() / ".claude" / "hooks" / "guard-destructive.py").exists()
    wf_dir = REPO / ".github" / "workflows"
    rails = sorted(p.name for p in wf_dir.glob("*merge*.yml")) if wf_dir.exists() else []
    gate: dict[str, str] = {}
    proj = REPO / "PROJECT.yaml"
    if proj.exists():
        txt = proj.read_text(encoding="utf-8")
        for key in ("test", "typecheck", "lint", "build", "auto_merge", "loop_enabled"):
            m = re.search(rf"^{key}\s*:\s*(.+)$", txt, re.M)
            if m:
                gate[key] = m.group(1).strip().strip("\"'")
    return {"guard": guard, "rails": rails, "gate": gate}


def gather() -> dict:
    prs = _gh(
        [
            "pr",
            "list",
            "--state",
            "open",
            "--json",
            "number,title,headRefName,mergeable,mergeStateStatus,isDraft",
        ]
    )
    runs = _gh(
        ["run", "list", "--limit", "10", "--json", "status,conclusion,headBranch,name,createdAt"]
    )
    latest: dict[str, dict] = {}
    for r in runs:
        b = r.get("headBranch")
        if b and b not in latest:
            latest[b] = r
    for pr in prs:
        r = latest.get(pr.get("headRefName", ""), {})
        pr["ci"] = r.get("conclusion") or r.get("status") or "none"
        pr["auto_merge"] = pr.get("headRefName", "").startswith(("loop/", "claude/"))
    issues = _gh(["issue", "list", "--state", "open", "--json", "number,title"])
    dirty = [x for x in _run(["git", "status", "--porcelain"]).splitlines() if x]
    return {
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "?",
        "clean": not dirty,
        "dirty_count": len(dirty),
        "main_log": _run(["git", "log", "origin/main", "--oneline", "-8"]).splitlines(),
        "prs": prs,
        "runs": runs,
        "issues": issues,
        "ladder": _parse_backlog(),
        "pipeline": _pipeline_state(),
    }


PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Forge-Mind Operator Console</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--bd:#30363d;--fg:#e6edf3;--mut:#8b949e;
--grn:#3fb950;--amb:#d29922;--red:#f85149;--blu:#58a6ff;--gry:#6e7681}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 ui-sans-serif,system-ui,Segoe UI,sans-serif}
header{display:flex;align-items:center;gap:14px;padding:14px 22px;border-bottom:1px solid var(--bd);
position:sticky;top:0;background:var(--bg);flex-wrap:wrap}
header h1{font-size:16px;margin:0;font-weight:700}
.pill{font-size:12px;padding:2px 9px;border-radius:999px;border:1px solid var(--bd);color:var(--mut)}
.pill.ok{color:var(--grn);border-color:#1f4429}.pill.bad{color:var(--red);border-color:#5c2321}
button{margin-left:auto;background:var(--card);color:var(--fg);border:1px solid var(--bd);
border-radius:7px;padding:6px 12px;cursor:pointer;font-size:13px}button:hover{border-color:var(--blu)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px;padding:18px 22px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:14px 16px}
.card h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);margin:0 0 10px}
.row{display:flex;gap:8px;align-items:baseline;padding:5px 0;border-top:1px solid #21262d}
.row:first-of-type{border-top:0}
.dot{width:9px;height:9px;border-radius:50%;flex:none;align-self:center}
.t-grn{background:var(--grn)}.t-amb{background:var(--amb)}.t-red{background:var(--red)}
.t-blu{background:var(--blu)}.t-gry{background:var(--gry)}
.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12.5px;color:var(--mut)}
.num{color:var(--blu);font-weight:600}.mut{color:var(--mut)}.done{color:var(--grn)}
a{color:var(--blu);text-decoration:none}.tag{font-size:11px;color:var(--mut);border:1px solid var(--bd);
border-radius:5px;padding:0 5px;margin-left:4px}.big{grid-column:1/-1}
.empty{color:var(--gry);font-style:italic}
</style></head><body>
<header><h1>🛠 Forge-Mind Operator Console</h1>
<span class=pill id=branch>branch</span><span class=pill id=clean>tree</span>
<span class=pill id=updated></span>
<button onclick=load()>Refresh</button></header>
<div class=grid id=grid>loading…</div>
<script>
const REPO="https://github.com/BeeJay805/Forge-Mind";
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function ciTone(s){if(s==='success')return'grn';if(s==='failure'||s==='cancelled')return'red';
if(s==='in_progress'||s==='queued'||s==='pending')return'amb';return'gry'}
function mergeTone(s){if(s==='CLEAN')return'grn';if(s==='UNSTABLE'||s==='BEHIND')return'amb';
if(s==='BLOCKED'||s==='DIRTY')return'red';return'gry'}
function card(title,inner){return `<div class=card><h2>${title}</h2>${inner||'<div class=empty>none</div>'}</div>`}
function rowsOf(arr,fn){return arr&&arr.length?arr.map(fn).join(''):'<div class=empty>none</div>'}
async function load(){
 const grid=document.getElementById('grid');
 let d;try{d=await(await fetch('/api/status',{cache:'no-store'})).json()}catch(e){grid.innerHTML='<div class=card>backend unreachable</div>';return}
 const b=document.getElementById('branch');b.textContent='⎇ '+d.branch;
 const c=document.getElementById('clean');c.textContent=d.clean?'working tree clean':d.dirty_count+' uncommitted';
 c.className='pill '+(d.clean?'ok':'bad');
 document.getElementById('updated').textContent='updated '+new Date().toLocaleTimeString();
 const p=d.pipeline||{},g=p.gate||{};
 const pipe=`<div class=row><span class="dot t-${p.guard?'grn':'red'}"></span>destructive-op guard <span class=mut>${p.guard?'active (hard wall)':'MISSING'}</span></div>`
  +`<div class=row><span class="dot t-${(p.rails||[]).length?'grn':'gry'}"></span>auto-merge rail <span class=mut>${(p.rails||[]).join(', ')||'none'}</span></div>`
  +`<div class=row><span class="dot t-blu"></span>auto-merge <span class=mut>${g.auto_merge||'?'} · loops ${g.loop_enabled||'?'}</span></div>`
  +(g.lint?`<div class=row><span class=mono>gate: ${esc(g.test||'')} · ${esc(g.typecheck||'')} · lint · ${esc(g.build||'')}</span></div>`:'');
 const prs=rowsOf(d.prs,pr=>`<div class=row><span class="dot t-${ciTone(pr.ci)}"></span>
  <span class=num>#${pr.number}</span> <a href="${REPO}/pull/${pr.number}" target=_blank>${esc(pr.title)}</a>
  <span class=tag>${esc(pr.headRefName)}</span>
  <span class=tag>CI ${esc(pr.ci)}</span><span class=tag>${esc(pr.mergeStateStatus||'')}</span>
  ${pr.auto_merge?'<span class=tag>auto-merge</span>':''}</div>`);
 const runs=rowsOf(d.runs,r=>`<div class=row><span class="dot t-${ciTone(r.conclusion||r.status)}"></span>
  <span class=mut>${esc(r.conclusion||r.status)}</span> <span class=mono>${esc(r.headBranch)}</span>
  <span class=tag>${esc(r.name)}</span></div>`);
 const issues=rowsOf(d.issues,i=>`<div class=row><span class="dot t-amb"></span>
  <span class=num>#${i.number}</span> <a href="${REPO}/issues/${i.number}" target=_blank>${esc(i.title)}</a></div>`);
 const log=rowsOf(d.main_log,l=>`<div class=row><span class=mono>${esc(l)}</span></div>`);
 const L=d.ladder||{};
 const lad=(arr,cls)=>rowsOf(arr,x=>`<div class=row><span class="dot t-${x.done?'grn':cls}"></span>
  <span class="${x.done?'done':''}">${esc(x.title)}</span></div>`);
 const ladder=`<div class=mut style=margin-bottom:6px>FEEDBACK (tops the queue)</div>${lad(L.feedback,'amb')}
  <div class=mut style=margin:10px_0_6px>NOW</div>${lad(L.now,'blu')}
  <div class=mut style=margin:10px_0_6px>NEXT</div>${lad(L.next,'gry')}
  <div class=mut style=margin:10px_0_6px>LATER (Fable-blocked)</div>${lad(L.later,'red')}
  <div class=mut style=margin:10px_0_6px>DONE (${(L.done||[]).length})</div>${lad((L.done||[]).slice(0,4),'grn')}`;
 grid.innerHTML=card('Pipeline & guard',pipe)+card('Open PRs',prs)
  +card('North Star ladder',ladder)+card('Recent CI runs',runs)
  +card('Open issues (alerts + feedback)',issues)
  +'<div class=big>'+card('Recent merges to main',log)+'</div>';
}
load();setInterval(load,20000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:  # quiet
        pass

    def do_GET(self) -> None:
        if self.path.startswith("/api/status"):
            body = json.dumps(gather()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Forge-Mind Operator Console -> http://localhost:{port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
