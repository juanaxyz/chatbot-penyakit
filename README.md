# Chatbot Penyakit (Nyx AI)

Chatbot asisten AI medis berbasis LLM lokal (Ollama) dan LangGraph ReAct Agent dengan kemampuan tool calling langsung ke database PostgreSQL (`capstone_paru`) tanpa memerlukan database vektor terpisah.

Dilengkapi web server FastAPI, antarmuka chat HTML sederhana untuk testing, serta CORS aktif untuk integrasi frontend eksternal (React, Vue, Next.js).

---

## 🚀 Fitur Utama

- **ReAct Agent**: Mengarahkan pertanyaan pengguna ke tool SQL yang sesuai secara dinamis.
- **6 Medical Tools (Native SQL)**:
  1. `search_disease_by_location`: Cari penyakit berdasarkan bagian tubuh/organ (`penyakit` + `bagian_tubuh`).
  2. `get_article_by_disease`: Ambil artikel edukasi medis lengkap (`artikel.konten_teks`).
  3. `get_disease_urgency`: Cek tingkat urgensi medis (`penyakit.tingkat_urgensi` - normal, waspada, darurat).
  4. `get_pathogen_by_disease`: Informasi mikroorganisme penyebab infeksi (`patogen`).
  5. `get_disease_references`: Link referensi medis resmi (`referensi.url`).
  6. `search_faq`: Tanya jawab umum seputar sistem & penggunaan fitur body map (`faq`).
- **Web UI & API Server**: FastAPI melayani testing UI di `GET /` dan REST API di `POST /chat`.

---

## 🛠️ Prasyarat

1. **Python 3.10+**
2. **Ollama**:
   ```bash
   ollama pull lfm3-omni:latest
   ```
3. **PostgreSQL** dengan database `capstone_paru`.

---

## ⚙️ Instalasi & Konfigurasi

1. Clone repositori:
   ```bash
   git clone https://github.com/juanaxyz/chatbot-penyakit.git
   cd chatbot-penyakit
   ```

2. Buat file konfigurasi database dari contoh:
   ```bash
   cp tools.yaml.example tools.yaml
   ```
   Sesuaikan `host`, `port`, `database`, `user`, dan `password` pada `tools.yaml`.

3. Install dependensi:
   ```bash
   pip install fastapi uvicorn psycopg pyyaml langchain langchain-ollama langgraph
   ```

---

## ▶️ Menjalankan Server

Jalankan server FastAPI:
```bash
python server.py
```
Akses antarmuka chat di browser:
👉 **http://127.0.0.1:8000**

Jalankan unit test untuk memastikan koneksi database dan seluruh 6 tool berfungsi:
```bash
python test_server.py
```

---

## 🔌 Integrasi Frontend

### Endpoint Chat: `POST /chat`

**Request:**
```json
{
  "message": "Apakah Pneumotoraks berbahaya dan perlu ke IGD?",
  "history": [
    {"role": "user", "content": "Halo Nyx"},
    {"role": "assistant", "content": "Halo! Ada yang bisa dibantu?"}
  ]
}
```

**Response:**
```json
{
  "reply": "Pneumotoraks adalah kondisi darurat (urgensi: darurat)...",
  "tool_calls": [
    {
      "name": "get_disease_urgency",
      "args": {"disease_name": "Pneumotoraks"}
    }
  ]
}
```

### Contoh Fetch (JavaScript):
```javascript
const response = await fetch("http://localhost:8000/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    message: "Penyakit apa saja di bagian dada?",
    history: []
  })
});
const data = await response.json();
console.log(data.reply);
```
