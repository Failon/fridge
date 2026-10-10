from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from .scanner import Scanner


def create_app(scanner: Scanner) -> FastAPI:
    app = FastAPI(title="SpreadCore")

    @app.get("/api/state")
    def state():
        return scanner.state_dict()

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        """Push the full state to the browser on every re-evaluation."""
        await sock.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=1)
        scanner.listeners.add(q)
        try:
            await sock.send_text(json.dumps(scanner.state_dict(), ensure_ascii=False))
            while True:
                await sock.send_text(await q.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            scanner.listeners.discard(q)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    return app


PAGE = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SpreadCore</title>
<style>
:root{--bg:#0e1116;--card:#171b22;--fg:#e6e6e6;--mut:#8b949e;--ok:#3fb950;--bad:#f85149;--acc:#58a6ff}
body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:12px 16px;border-bottom:1px solid #222}
h1{font-size:18px;margin:0}.mut{color:var(--mut)}input{background:var(--card);color:var(--fg);border:1px solid #333;border-radius:6px;padding:4px 8px;width:90px}
.books span{margin-right:10px}.err{color:var(--bad)}main{padding:16px;display:grid;gap:12px}
.card{background:var(--card);border-radius:10px;padding:12px 14px}.p{font-size:20px;font-weight:700;color:var(--ok)}
table{width:100%;border-collapse:collapse;margin-top:8px}td{padding:4px 6px;border-top:1px solid #222}a{color:var(--acc)}
</style></head><body>
<header><h1>SpreadCore</h1><span class="books" id="books"></span>
<label class="mut">Банк $ <input id="bank" type="number" value="100"></label><span class="mut" id="upd"></span><b id="live"></b></header>
<main id="list"><div class="mut">Загрузка…</div></main>
<script>
const N={stake:'Stake',parimatch:'Parimatch',goated:'Goated',polymarket:'Polymarket'};
const bank=document.getElementById('bank');try{bank.value=localStorage.bank||100}catch(e){}
bank.oninput=()=>{try{localStorage.bank=bank.value}catch(e){};render()};let S=null;
async function load(){try{S=await (await fetch('api/state')).json();render()}catch(e){}}
let poll=null;function connect(){const u=new URL('ws',location.href);u.protocol=u.protocol.replace('http','ws');
const ws=new WebSocket(u);ws.onopen=()=>{clearInterval(poll);poll=null;live(true)};
ws.onmessage=e=>{S=JSON.parse(e.data);render()};
ws.onclose=()=>{live(false);if(!poll)poll=setInterval(load,3000);setTimeout(connect,2000)}}
function live(on){document.getElementById('live').textContent=on?'● live':'○ offline';document.getElementById('live').style.color=on?'var(--ok)':'var(--bad)'}
function esc(s){return String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function render(){if(!S)return;const B=+bank.value||100;
document.getElementById('books').innerHTML=Object.entries(S.counts).map(([k,v])=>
`<span class="${S.errors[k]?'err':''}" title="${esc(S.errors[k]||'')}">${N[k]||k}: ${v}</span>`).join('')+
`<span class="mut">пар: ${S.matched} · тиков: ${S.ticks} · WS: ${Object.keys(S.streams).map(k=>N[k]||k).join(', ')||'—'}</span>`;
document.getElementById('upd').textContent=S.updated?'обновлено '+new Date(S.updated).toLocaleTimeString()+'.'+String(new Date(S.updated).getMilliseconds()).padStart(3,'0'):'';
const L=document.getElementById('list');
if(!S.arbs.length){L.innerHTML='<div class="mut">Вилок сейчас нет</div>';return}
L.innerHTML=S.arbs.map(a=>{const inv=a.legs.reduce((s,l)=>s+1/l.odds,0);
return `<div class="card"><span class="p">${a.profit.toFixed(2)}%</span> <b>${esc(a.title)}</b>
<span class="mut">${esc(a.sport)} · ${esc(a.league)} · ${a.market} · ${a.start?new Date(a.start).toLocaleString():''}</span>
<table>${a.legs.map(l=>`<tr><td>${N[l.book]||l.book}</td><td>${esc(l.team)}</td><td>${l.odds}</td>
<td><b>$${(B/(l.odds*inv)).toFixed(2)}</b></td><td class="mut">${l.limit?'лимит $'+Math.round(l.limit):''}</td>
<td>${l.url?`<a href="${esc(l.url)}" target="_blank" rel="noopener">открыть</a>`:''}</td></tr>`).join('')}</table>
<div class="mut">прибыль ≈ $${(B/inv-B).toFixed(2)}</div></div>`}).join('')}
load();connect();
</script></body></html>"""
