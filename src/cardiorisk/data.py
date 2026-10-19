from pathlib import Path
import pandas as pd
EXPECTED=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]
def load_data(path="data/processed/heart_disease_clean.csv"):
    df=pd.read_csv(Path(path))
    missing=set(EXPECTED)-set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    if not set(df.target.unique()).issubset({0,1}): raise ValueError("Target must be binary")
    return df
