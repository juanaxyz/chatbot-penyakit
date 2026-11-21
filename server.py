import json
import yaml
import psycopg
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# 1. Load DB Config
# ponytail: sync single connection per request ceiling; upgrade to psycopg-pool if RPS > 50
with open("tools.yaml") as f:
    db_cfg = yaml.safe_load(f)["sources"]["disease-pg-source"]


def query_db(sql: str, params: tuple) -> list[tuple]:
    """Jalankan query read-only ke PostgreSQL dan kembalikan seluruh baris."""
    # ponytail: sync single connection per request ceiling; upgrade to psycopg-pool if RPS > 50
    with psycopg.connect(
        host=db_cfg["host"],
        port=db_cfg["port"],
        dbname=db_cfg["database"],
        user=db_cfg["user"],
        password=db_cfg["password"],
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


@tool
def search_disease_by_location(location: str) -> str:
    """Cari daftar penyakit berdasarkan nama bagian tubuh (misal: dada, paru)."""
    loc = location.strip()
    rows = query_db(
        """
        SELECT p.nama, p.ringkasan, bt.nama AS nama_bagian_tubuh
        FROM penyakit p
        JOIN penyakit_bagian_tubuh pbt ON p.id = pbt.id_penyakit
        JOIN bagian_tubuh bt ON pbt.id_bagian_tubuh = bt.id
        WHERE bt.nama ILIKE %s
        """,
        (f"%{loc}%",),
    )
    data = [
        {"nama": r[0], "ringkasan": r[1], "bagian_tubuh": r[2]} for r in rows
    ]

    if not data:
        return f"Tidak ditemukan penyakit untuk bagian tubuh '{loc}' di database."
    return json.dumps(data[:8], ensure_ascii=False)


@tool
def get_article_by_disease(disease_name: str) -> str:
    """Ambil artikel edukasi lengkap (gejala, penyebab, pengobatan) berdasarkan nama penyakit (misal: Bronkitis, Tuberkulosis)."""
    name = disease_name.strip()
    rows = query_db(
        """
        SELECT a.id, a.konten_teks
        FROM artikel a
        JOIN penyakit p ON a.id_penyakit = p.id
        WHERE p.nama ILIKE %s
        """,
        (f"%{name}%",),
    )

    if not rows:
        return f"Tidak ditemukan artikel untuk penyakit '{name}' di database."
    # ponytail: first-article-only ceiling; upgrade ke top-N scoring saat > 1 artikel per penyakit
    return json.dumps(
        {"id": rows[0][0], "konten": rows[0][1]}, ensure_ascii=False
    )


@tool
def get_disease_urgency(disease_name: str) -> str:
    """Cek tingkat urgensi medis suatu penyakit (normal, waspada, atau darurat/butuh IGD segera)."""
    name = disease_name.strip()
    rows = query_db(
        """
        SELECT nama, tingkat_urgensi, ringkasan
        FROM penyakit
        WHERE nama ILIKE %s
        """,
        (f"%{name}%",),
    )
    if not rows:
        return f"Data urgensi untuk penyakit '{name}' tidak ditemukan."
    return json.dumps(
        {"nama": rows[0][0], "urgensi": rows[0][1], "ringkasan": rows[0][2]},
        ensure_ascii=False,
    )


@tool
def get_pathogen_by_disease(disease_name: str) -> str:
    """Ambil informasi patogen atau mikroorganisme penyebab penyakit (virus, bakteri, jamur)."""
    name = disease_name.strip()
    rows = query_db(
        """
        SELECT pt.nama AS patogen, pt.jenis, pt.deskripsi
        FROM penyakit_patogen pp
        JOIN penyakit p ON pp.id_penyakit = p.id
        JOIN patogen pt ON pp.id_patogen = pt.id
        WHERE p.nama ILIKE %s
        """,
        (f"%{name}%",),
    )
    if not rows:
        return f"Tidak ditemukan data patogen penyebab untuk penyakit '{name}'."
    data = [{"patogen": r[0], "jenis": r[1], "deskripsi": r[2]} for r in rows]
    return json.dumps(data, ensure_ascii=False)


@tool
def get_disease_references(disease_name: str) -> str:
    """Ambil link sumber rujukan dan referensi medis resmi untuk suatu penyakit."""
    name = disease_name.strip()
    rows = query_db(
        """
        SELECT r.url
        FROM referensi r
        JOIN penyakit p ON r.id_penyakit = p.id
        WHERE p.nama ILIKE %s
        """,
        (f"%{name}%",),
    )
    if not rows:
        return f"Tidak ditemukan link referensi untuk penyakit '{name}'."
    return json.dumps([r[0] for r in rows], ensure_ascii=False)


@tool
def search_faq(keyword: str) -> str:
    """Cari jawaban pertanyaan umum seputar sistem, cara pakai peta tubuh (body map), atau alur pemeriksaan."""
    kw = keyword.strip()
    rows = query_db(
        """
        SELECT pertanyaan, jawaban
        FROM faq
        WHERE pertanyaan ILIKE %s OR jawaban ILIKE %s
        LIMIT 3
        """,
        (f"%{kw}%", f"%{kw}%"),
    )
    if not rows:
        return f"Tidak ditemukan FAQ terkait '{kw}'."
    data = [{"pertanyaan": r[0].strip(), "jawaban": r[1].strip()} for r in rows]
    return json.dumps(data, ensure_ascii=False)


# 2. LLM & Agent
SYSTEM_PROMPT = """Nama kamu adalah Nyx, asisten AI informasi penyakit dan kesehatan paru.
PANDUAN PEMILIHAN TOOL:
1. `search_disease_by_location`: Gunakan saat user menanyakan penyakit berdasarkan organ/bagian tubuh (dada, paru, dll).
2. `get_article_by_disease`: Gunakan saat user ingin edukasi komprehensif (gejala, pengobatan, pencegahan) suatu penyakit.
3. `get_disease_urgency`: Gunakan saat user bertanya tingkat bahaya/urgensi, apakah darurat/butuh IGD, atau seberapa parah suatu penyakit.
4. `get_pathogen_by_disease`: Gunakan saat user bertanya penyebab infeksi, virus, bakteri, atau penularan penyakit.
5. `get_disease_references`: Gunakan saat user meminta link/sumber rujukan medis resmi.
6. `search_faq`: Gunakan saat user bertanya cara pakai website, fitur body map/peta tubuh, atau pertanyaan umum sistem.

ATURAN MEDIS:
- Gunakan data database sebagai sumber utama kebenaran. Jangan mengarang data medis.
- Jangan pernah memberikan diagnosis medis pasti. Selalu anjurkan konsultasi ke dokter spesialis."""

llm = ChatOllama(model="lfm3-omni:latest", temperature=0.7)
all_tools = [
    search_disease_by_location,
    get_article_by_disease,
    get_disease_urgency,
    get_pathogen_by_disease,
    get_disease_references,
    search_faq,
]
agent = create_react_agent(llm, tools=all_tools, prompt=SYSTEM_PROMPT)

# 3. FastAPI App
app = FastAPI(title="Nyx Disease Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []


HTML_PAGE = """<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Nyx - Chatbot Penyakit</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
    body { background: #0f172a; color: #e2e8f0; display: flex; justify-content: center; height: 100vh; padding: 16px; }
    .container { width: 100%; max-width: 760px; display: flex; flex-direction: column; background: #1e293b; border-radius: 12px; border: 1px solid #334155; overflow: hidden; }
    header { padding: 16px 20px; background: #0f172a; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }
    header h1 { font-size: 1.1rem; color: #38bdf8; font-weight: 600; }
    header .status { font-size: 0.8rem; color: #4ade80; display: flex; align-items: center; gap: 6px; }
    header .dot { width: 8px; height: 8px; background: #4ade80; border-radius: 50%; }
    .chat-box { flex: 1; padding: 16px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }
    .msg { max-width: 80%; padding: 10px 14px; border-radius: 10px; font-size: 0.92rem; line-height: 1.45; white-space: pre-wrap; word-break: break-word; }
    .msg.user { align-self: flex-end; background: #2563eb; color: #fff; }
    .msg.bot { align-self: flex-start; background: #334155; color: #f1f5f9; border: 1px solid #475569; }
    .tool-badge { font-size: 0.75rem; background: #0369a1; color: #bae6fd; padding: 3px 8px; border-radius: 4px; margin-bottom: 6px; display: inline-block; }
    .suggestions { display: flex; gap: 8px; padding: 8px 16px; background: #1e293b; border-top: 1px solid #334155; overflow-x: auto; }
    .chip { background: #334155; color: #94a3b8; border: none; padding: 6px 12px; border-radius: 16px; font-size: 0.8rem; cursor: pointer; white-space: nowrap; }
    .chip:hover { background: #475569; color: #e2e8f0; }
    form { display: flex; padding: 12px 16px; background: #0f172a; gap: 8px; border-top: 1px solid #334155; }
    input { flex: 1; padding: 10px 14px; background: #1e293b; border: 1px solid #334155; border-radius: 8px; color: #fff; font-size: 0.95rem; outline: none; }
    input:focus { border-color: #38bdf8; }
    button { padding: 10px 18px; background: #0284c7; color: #fff; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Nyx - Chatbot Penyakit (capstone_paru)</h1>
      <div class="status"><span class="dot"></span> Online</div>
    </header>
    <div id="chatBox" class="chat-box">
      <div class="msg bot">Halo! Saya Nyx. Anda bisa menanyakan daftar penyakit berdasarkan bagian tubuh (misalnya dada atau paru). Ada yang bisa dibantu?</div>
    </div>
    <div class="suggestions">
      <button class="chip" onclick="quickSend('Penyakit apa saja di bagian dada?')">Penyakit di dada</button>
      <button class="chip" onclick="quickSend('Apakah Pneumotoraks berbahaya dan perlu ke IGD?')">Cek Urgensi</button>
      <button class="chip" onclick="quickSend('Apa virus atau bakteri penyebab COVID-19?')">Cek Patogen</button>
      <button class="chip" onclick="quickSend('Minta link referensi artikel Tuberkulosis')">Referensi Medis</button>
      <button class="chip" onclick="quickSend('Bagaimana cara menggunakan fitur body map?')">Tanya FAQ</button>
    </div>
    <form id="chatForm">
      <input id="userInput" type="text" placeholder="Ketik pertanyaan tentang penyakit / bagian tubuh..." autocomplete="off" required>
      <button id="sendBtn" type="submit">Kirim</button>
    </form>
  </div>
  <script>
    const chatBox = document.getElementById('chatBox');
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    // ponytail: client-side memory history ceiling; upgrade to backend session ID when multi-user auth added
    const history = [];

    function appendMsg(role, text, toolCalls = []) {
      const div = document.createElement('div');
      div.className = `msg ${role}`;
      if (toolCalls && toolCalls.length > 0) {
        const badge = document.createElement('div');
        badge.className = 'tool-badge';
        badge.innerText = `Tool: ${toolCalls.map(t => `${t.name}(${JSON.stringify(t.args)})`).join(', ')}`;
        div.appendChild(badge);
      }
      const span = document.createElement('div');
      span.innerText = text;
      div.appendChild(span);
      chatBox.appendChild(div);
      chatBox.scrollTop = chatBox.scrollHeight;
    }

    async function send(text) {
      if (!text.trim()) return;
      appendMsg('user', text);
      userInput.value = '';
      userInput.disabled = true;
      sendBtn.disabled = true;

      const loadingDiv = document.createElement('div');
      loadingDiv.className = 'msg bot';
      loadingDiv.innerText = 'Sedang mencari data...';
      chatBox.appendChild(loadingDiv);
      chatBox.scrollTop = chatBox.scrollHeight;

      try {
        const res = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, history: history })
        });
        const data = await res.json();
        chatBox.removeChild(loadingDiv);
        appendMsg('bot', data.reply, data.tool_calls);
        history.push({ role: 'user', content: text });
        history.push({ role: 'assistant', content: data.reply });
      } catch (err) {
        chatBox.removeChild(loadingDiv);
        appendMsg('bot', 'Error: Gagal menghubungi server chatbot.');
      } finally {
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
      }
    }

    chatForm.addEventListener('submit', (e) => {
      e.preventDefault();
      send(userInput.value);
    });

    function quickSend(txt) {
      userInput.value = txt;
      send(txt);
    }
  </script>
</body>
</html>
"""


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


@app.post("/chat")
async def chat(req: ChatRequest):
    # ponytail: in-memory messages list ceiling; upgrade to LangGraph checkpointer when persistent threads needed
    messages = []
    for h in req.history[-6:]:
        messages.append((h.role if h.role == "user" else "ai", h.content))
    messages.append(("user", req.message))

    res = await agent.ainvoke({"messages": messages})
    tool_calls = []
    for m in res["messages"]:
        tc = getattr(m, "tool_calls", None)
        if tc:
            tool_calls.extend(tc)

    reply = res["messages"][-1].content
    return {"reply": reply, "tool_calls": tool_calls}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
