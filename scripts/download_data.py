from pathlib import Path
import hashlib
import pandas as pd
import urllib.request

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMNS = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num"]
raw = Path("data/raw/processed.cleveland.data")
clean = Path("data/processed/heart_disease_clean.csv")
raw.parent.mkdir(parents=True, exist_ok=True); clean.parent.mkdir(parents=True, exist_ok=True)
req = urllib.request.Request(URL, headers={"User-Agent":"Mozilla/5.0 cardiorisk-educational-project"})
with urllib.request.urlopen(req, timeout=60) as response:
    raw.write_bytes(response.read())
df = pd.read_csv(raw, header=None, names=COLUMNS, na_values="?")
for col in COLUMNS: df[col] = pd.to_numeric(df[col], errors="coerce")
df["target"] = (df.pop("num") > 0).astype(int)
df.to_csv(clean, index=False)
print(f"Rows={len(df)}, columns={len(df.columns)}, missing={int(df.isna().sum().sum())}")
print("SHA256:", hashlib.sha256(clean.read_bytes()).hexdigest())
