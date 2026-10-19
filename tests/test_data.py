import pandas as pd
from src.cardiorisk.data import load_data, EXPECTED
def test_load_data(tmp_path):
    p=tmp_path/"x.csv"; pd.DataFrame([[50,1,2,120,200,0,0,150,0,1.0,2,0,3,1]],columns=EXPECTED).to_csv(p,index=False)
    assert load_data(p).shape==(1,14)
def test_reject_nonbinary(tmp_path):
    p=tmp_path/"x.csv"; df=pd.DataFrame([[50,1,2,120,200,0,0,150,0,1.0,2,0,3,4]],columns=EXPECTED); df.to_csv(p,index=False)
    try: load_data(p); assert False
    except ValueError: assert True
