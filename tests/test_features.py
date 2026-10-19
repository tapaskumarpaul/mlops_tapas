import pandas as pd
from src.cardiorisk.features import build_preprocessor

def test_preprocessor_handles_missing():
    X=pd.DataFrame([{"age":54,"sex":1,"cp":4,"trestbps":140,"chol":239,"fbs":0,"restecg":0,"thalach":160,"exang":0,"oldpeak":1.2,"slope":2,"ca":None,"thal":3},{"age":60,"sex":0,"cp":3,"trestbps":130,"chol":250,"fbs":1,"restecg":1,"thalach":145,"exang":1,"oldpeak":2.0,"slope":1,"ca":1,"thal":7}])
    z=build_preprocessor().fit_transform(X); assert z.shape[0]==2; assert not pd.isna(z).any()
