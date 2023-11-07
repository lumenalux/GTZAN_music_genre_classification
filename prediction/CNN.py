import numpy as np

def get_cnn_prediction(cnn_model, features):
        return np.argmax(cnn_model.predict(features), axis=1)