from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
NUMERIC_FEATURES=["age","trestbps","chol","thalach","oldpeak"]
CATEGORICAL_FEATURES=["sex","cp","fbs","restecg","exang","slope","ca","thal"]
def build_preprocessor():
    num=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
    cat=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])
    return ColumnTransformer([("num",num,NUMERIC_FEATURES),("cat",cat,CATEGORICAL_FEATURES)])
