import pandas as pd
from src.cardiorisk.model import make_model
def test_model_predicts_probability():
    rows=[]
    for i in range(20): rows.append({"age":40+i,"sex":i%2,"cp":1+i%4,"trestbps":110+i,"chol":180+i*3,"fbs":i%2,"restecg":i%3,"thalach":170-i,"exang":i%2,"oldpeak":i/10,"slope":1+i%3,"ca":i%4,"thal":3 if i%2==0 else 7})
    X=pd.DataFrame(rows); y=[i%2 for i in range(20)]; m=make_model("logistic_regression").fit(X,y); p=m.predict_proba(X)[:,1]
    assert ((p>=0)&(p<=1)).all()
