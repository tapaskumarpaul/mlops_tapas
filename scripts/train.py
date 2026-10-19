from pathlib import Path
import json, joblib, mlflow, mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
from src.cardiorisk.data import load_data
from src.cardiorisk.model import make_model

df=load_data(); X=df.drop(columns="target"); y=df.target
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
cv=StratifiedKFold(5,shuffle=True,random_state=42)
Path("models").mkdir(exist_ok=True); Path("reports/figures").mkdir(parents=True,exist_ok=True)
mlflow.set_experiment("cardiorisk-heart-disease")
results=[]
for name in ["logistic_regression","random_forest"]:
    model=make_model(name)
    scores=cross_validate(model,Xtr,ytr,cv=cv,scoring=["accuracy","precision","recall","f1","roc_auc"])
    model.fit(Xtr,ytr); pred=model.predict(Xte); prob=model.predict_proba(Xte)[:,1]
    metrics={"cv_roc_auc_mean":scores["test_roc_auc"].mean(),"cv_roc_auc_std":scores["test_roc_auc"].std(),"test_accuracy":accuracy_score(yte,pred),"test_precision":precision_score(yte,pred),"test_recall":recall_score(yte,pred),"test_f1":f1_score(yte,pred),"test_roc_auc":roc_auc_score(yte,prob)}
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model",name); mlflow.log_param("cv_folds",5); mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model,"model",input_example=Xte.head(2))
    results.append((metrics["cv_roc_auc_mean"],name,model,metrics,pred,prob))
results.sort(key=lambda x:x[0],reverse=True); _,name,best,metrics,pred,prob=results[0]
joblib.dump(best,"models/heart_disease_pipeline.joblib")
Path("models/metadata.json").write_text(json.dumps({"model_name":name,"version":"1.0.0","metrics":metrics},indent=2))
ConfusionMatrixDisplay.from_predictions(yte,pred); plt.tight_layout(); plt.savefig("reports/figures/confusion_matrix.png",dpi=160); plt.close()
RocCurveDisplay.from_predictions(yte,prob); plt.tight_layout(); plt.savefig("reports/figures/roc_curve.png",dpi=160); plt.close()
print(json.dumps({"selected_model":name,**metrics},indent=2))
