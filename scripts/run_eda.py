from pathlib import Path
import matplotlib.pyplot as plt, seaborn as sns
from src.cardiorisk.data import load_data
out=Path("reports/figures"); out.mkdir(parents=True,exist_ok=True)
df=load_data()
sns.countplot(data=df,x="target"); plt.title("Class balance"); plt.tight_layout(); plt.savefig(out/"class_balance.png",dpi=160); plt.close()
df[["age","trestbps","chol","thalach","oldpeak"]].hist(figsize=(10,6)); plt.tight_layout(); plt.savefig(out/"histograms.png",dpi=160); plt.close()
sns.heatmap(df.corr(numeric_only=True),cmap="coolwarm",center=0); plt.title("Feature correlation"); plt.tight_layout(); plt.savefig(out/"correlation_heatmap.png",dpi=160); plt.close()
print(df.describe(include="all")); print("Missing values:
",df.isna().sum())
