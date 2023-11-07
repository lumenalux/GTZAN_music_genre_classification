import numpy as np
from xgboost import DMatrix

def get_xgb_prediction(xgb_model, features):
        return np.argmax(xgb_model.predict(DMatrix(data=features)), axis=1)