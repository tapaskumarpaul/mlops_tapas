from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from .features import build_preprocessor
def make_model(name, random_state=42):
    if name=="logistic_regression": clf=LogisticRegression(max_iter=2000,class_weight="balanced",random_state=random_state)
    elif name=="random_forest": clf=RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=random_state,n_jobs=-1)
    else: raise ValueError(f"Unsupported model: {name}")
    return Pipeline([("preprocessor",build_preprocessor()),("classifier",clf)])
