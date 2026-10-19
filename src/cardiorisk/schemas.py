from pydantic import BaseModel, Field
class PatientFeatures(BaseModel):
    age: float=Field(ge=18,le=120); sex:int=Field(ge=0,le=1); cp:int=Field(ge=1,le=4)
    trestbps:float=Field(ge=50,le=260); chol:float=Field(ge=80,le=700); fbs:int=Field(ge=0,le=1)
    restecg:int=Field(ge=0,le=2); thalach:float=Field(ge=40,le=250); exang:int=Field(ge=0,le=1)
    oldpeak:float=Field(ge=-3,le=10); slope:int=Field(ge=1,le=3); ca:float|None=Field(default=None,ge=0,le=4); thal:float|None=Field(default=None)
