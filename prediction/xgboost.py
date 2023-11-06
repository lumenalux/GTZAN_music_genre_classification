
from xgboost import DMatrix

def get_xgb_prediction(xgb_model, features):
        return xgb_model.predict(DMatrix(data=features)).to_list()