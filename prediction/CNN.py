import numpy as np

def get_cnn_prediction(cnn_model, features):
        return cnn_model.predict(features).tolist()[0]