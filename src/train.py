from pathlib import Path
import pandas as pd
import spacy
from spacy.training import Example

# 1. Muat & Bersihkan Data CSV
data_dir = Path("use_cases/pnm_study_case/sample")
dfs = []
for file in sorted(data_dir.glob("*.csv")):
    df = pd.read_csv(file, sep=";", dtype=str, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    if {"title", "body", "language"}.issubset(df.columns):
        dfs.append(df[["title", "body", "language"]])

data = pd.concat(dfs, ignore_index=True)
clean = lambda s: s.astype(str).str.strip("' \t\r\n")
data["language"] = clean(data["language"]).str.lower()
data = data[data["language"] == "en"].dropna(subset=["title", "body"])
texts = (clean(data["title"]) + "\n" + clean(data["body"])).drop_duplicates().tolist()
print(f"Total artikel bersih & unik: {len(texts)}")

# 2. Siapkan Anotasi Otomatis (PERSON, ORG, GPE, LOC)
nlp = spacy.load("en_core_web_sm")
LABELS = {"PERSON", "ORG", "GPE", "LOC"}
examples = []

for text in texts[:400]:
    doc = nlp(text[:5000])
    ents = [(e.start_char, e.end_char, e.label_) for e in doc.ents if e.label_ in LABELS]
    try:
        examples.append(Example.from_dict(nlp.make_doc(text[:5000]), {"entities": ents}))
    except ValueError:
        continue

print(f"Jumlah sampel training: {len(examples)}")

# 3. Latih Model NER
other_pipes = [p for p in nlp.pipe_names if p != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.resume_training()
    for epoch in range(10):
        losses = {}
        for i in range(0, len(examples), 16):
            nlp.update(examples[i:i + 16], sgd=optimizer, drop=0.2, losses=losses)
        print(f"Epoch {epoch + 1}/10 - Loss: {losses.get('ner', 0.0):.4f}")

# 4. Simpan Model
out_path = Path("models/ner_custom")
out_path.parent.mkdir(parents=True, exist_ok=True)
nlp.to_disk(out_path)
print(f"Model berhasil disimpan di: {out_path}")
