# Article Named Entity Recognition (NER) Service

Proyek pengolahan data teks dan ekstraksi Named Entity Recognition (NER) berbasis spaCy dan FastAPI untuk mendeteksi entitas penting seperti **PERSON**, **ORG**, **GPE**, dan **LOC** pada artikel berita.

---

## 📌 Fitur Utama
- **Pipeline Data & Training Ringkas (`src/train.py`)**: Memuat raw CSV, membersihkan data, filter artikel bahasa Inggris, deduplikasi, pseudo-annotation otomatis, dan melatih custom model spaCy NER.
- **Web UI & REST API Terpadu (`src/app.py`)**: Antarmuka web modern langsung di browser tanpa konfigurasi frontend terpisah, serta REST API untuk integrasi sistem.

---

## 📁 Struktur Proyek
```text
.
├── src/
│   ├── train.py              # Pipeline pembersihan data & training model NER
│   └── app.py                # Server FastAPI & Web Interface (Single-file UI)
├── models/
│   └── ner_custom/           # Model spaCy NER hasil pelatihan
├── use_cases/
│   └── pnm_study_case/
│       └── sample/           # Dataset mentah berupa file CSV
├── requirements.txt          # Dependensi Python
├── .gitignore                # File & direktori yang diabaikan Git
└── README.md                 # Dokumentasi proyek
```

---

## ⚙️ Instalasi & Persiapan

1. **Clone Repositori**:
   ```bash
   git clone <URL_REPOSITORI_ANDA>
   cd laprak
   ```

2. **Buat & Aktifkan Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # Di Linux / macOS
   # .venv\Scripts\activate     # Di Windows
   ```

3. **Install Dependensi**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Penggunaan

### 1. Pelatihan Model (Training)
Untuk menjalankan pembersihan data, pseudo-labeling, dan melatih model NER:
```bash
python src/train.py
```
Model hasil training akan disimpan di direktori `models/ner_custom`.

### 2. Menjalankan Web UI & API
Jalankan server aplikasi:
```bash
python src/app.py
```
atau menggunakan `uvicorn`:
```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

Buka peramban (browser) di:
- **Web Interface**: [http://localhost:8000](http://localhost:8000)
- **API Documentation (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔌 API Endpoint

### `POST /api/ner`
Mendeteksi entitas dari teks atau artikel.

**Request Body:**
```json
{
  "title": "Maybank Announcement",
  "body": "John Doe announced new financial policies in Kuala Lumpur."
}
```

**Response:**
```json
{
  "text": "Maybank Announcement\nJohn Doe announced new financial policies in Kuala Lumpur.",
  "entities": [
    {
      "text": "John Doe",
      "label": "PERSON",
      "start": 21,
      "end": 29
    },
    {
      "text": "Kuala Lumpur",
      "label": "GPE",
      "start": 59,
      "end": 71
    }
  ]
}
```

### Label Entitas yang Didukung:
| Label | Deskripsi |
|---|---|
| `PERSON` | Tokoh, orang, atau figur publik |
| `ORG` | Perusahaan, institusi, agensi, atau organisasi |
| `GPE` | Negara, kota, atau wilayah administratif |
| `LOC` | Lokasi non-GPE, bentang alam, pegunungan, perairan |
