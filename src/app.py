from pathlib import Path
import spacy
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Article NER Service")
MODEL_PATH = Path("models/ner_custom")
nlp = spacy.load(MODEL_PATH if MODEL_PATH.exists() else "en_core_web_sm")

HTML_PAGE = """<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Article NER Service</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; color: #1e293b; background: #f8fafc; }
    h1 { margin-bottom: 6px; }
    p.sub { color: #64748b; margin-top: 0; }
    input, textarea, button { width: 100%; box-sizing: border-box; padding: 10px; margin: 8px 0 16px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 14px; }
    textarea { min-height: 140px; font-family: inherit; }
    button { background: #2563eb; color: white; border: none; font-weight: 600; cursor: pointer; transition: background 0.2s; }
    button:hover { background: #1d4ed8; }
    table { width: 100%; border-collapse: collapse; margin-top: 16px; background: white; border-radius: 6px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    th, td { border: 1px solid #e2e8f0; padding: 10px; text-align: left; }
    th { background: #f1f5f9; font-weight: 600; }
    .badge { padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; background: #e0f2fe; color: #0369a1; }
  </style>
</head>
<body>
  <h1>Article NER Service</h1>
  <p class="sub">Ekstraksi Named Entity Recognition (PERSON, ORG, GPE, LOC) dari artikel.</p>
  <form id="form">
    <label><b>Judul:</b></label>
    <input id="title" placeholder="Masukkan judul artikel...">
    <label><b>Isi Artikel:</b></label>
    <textarea id="body" required placeholder="Masukkan isi artikel..."></textarea>
    <button type="submit" id="btn">Deteksi Entitas</button>
  </form>
  <div id="status" style="margin: 10px 0; color: #475569;"></div>
  <table id="tbl" style="display:none;">
    <thead><tr><th>Entitas</th><th>Label</th><th>Start</th><th>End</th></tr></thead>
    <tbody id="rows"></tbody>
  </table>
  <script>
    document.getElementById("form").onsubmit = async (e) => {
      e.preventDefault();
      const btn = document.getElementById("btn");
      const status = document.getElementById("status");
      const tbl = document.getElementById("tbl");
      const rows = document.getElementById("rows");
      btn.disabled = true; status.innerText = "Memproses...";
      rows.innerHTML = "";
      try {
        const res = await fetch("/api/ner", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            title: document.getElementById("title").value,
            body: document.getElementById("body").value
          })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Gagal memproses");
        status.innerText = `Ditemukan ${data.entities.length} entitas.`;
        tbl.style.display = data.entities.length ? "table" : "none";
        data.entities.forEach(ent => {
          rows.innerHTML += `<tr><td><b>${ent.text}</b></td><td><span class="badge">${ent.label}</span></td><td>${ent.start}</td><td>${ent.end}</td></tr>`;
        });
      } catch (err) {
        status.innerText = "Error: " + err.message;
      } finally {
        btn.disabled = false;
      }
    };
  </script>
</body>
</html>"""

class ArticleRequest(BaseModel):
    title: str = ""
    body: str

@app.get("/", response_class=HTMLResponse)
@app.get("/panel", response_class=HTMLResponse)
def index():
    return HTML_PAGE

@app.get("/health")
def health():
    return {"status": "ok", "model": str(MODEL_PATH if MODEL_PATH.exists() else "en_core_web_sm")}

@app.post("/api/ner")
def extract_entities(req: ArticleRequest):
    text = f"{req.title.strip()}\n{req.body.strip()}".strip()
    if not text:
        raise HTTPException(status_code=400, detail="Judul atau isi artikel wajib diisi.")
    doc = nlp(text)
    return {
        "text": text,
        "entities": [
            {"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char}
            for e in doc.ents
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
