from pathlib import Path
import json, logging, time, uuid, joblib, pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from .schemas import PatientFeatures
logging.basicConfig(level=logging.INFO,format="%(message)s"); log=logging.getLogger("cardiorisk")
app=FastAPI(title="CardioRisk Educational API",version="1.0.0")
REQ=Counter("cardiorisk_requests_total","Prediction requests"); ERR=Counter("cardiorisk_prediction_errors_total","Prediction errors"); LAT=Histogram("cardiorisk_request_latency_seconds","Prediction latency")
MODEL_PATH=Path("models/heart_disease_pipeline.joblib"); model=joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
@app.get("/health")
def health(): return {"status":"ok" if model is not None else "model_not_loaded","model_version":"1.0.0"}
@app.get("/metrics")
def metrics(): return Response(generate_latest(),media_type=CONTENT_TYPE_LATEST)
@app.post("/predict")
def predict(x:PatientFeatures,request:Request):
    if model is None: raise HTTPException(503,"Model is not loaded. Run scripts/train.py first.")
    rid=str(uuid.uuid4()); start=time.perf_counter(); REQ.inc()
    try:
        row=pd.DataFrame([x.model_dump()]); p=float(model.predict_proba(row)[0,1]); pred=int(p>=.5)
        return {"prediction":pred,"risk_label":"higher_model_estimated_risk" if pred else "lower_model_estimated_risk","confidence":round(p,4),"model_version":"1.0.0","request_id":rid}
    except Exception as e: ERR.inc(); raise HTTPException(500,str(e))
    finally:
        latency=time.perf_counter()-start; LAT.observe(latency); log.info(json.dumps({"request_id":rid,"endpoint":"/predict","latency_ms":round(latency*1000,2)}))
