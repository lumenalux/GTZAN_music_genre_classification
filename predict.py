from functools import partial

from flask import Flask, request, jsonify
import librosa
import numpy as np
import joblib
import tensorflow as tf
from xgboost import Booster, DMatrix

from extract import extract_features, prepare_features_for_model
from prediction.xgboost import get_xgb_prediction
from prediction.CNN import get_cnn_prediction
from prediction.random_forest import get_random_forest_prediction
from prediction.KNN import get_knn_prediction
from prediction.deep_learning import get_dp_prediction, get_dp_ensemble_prediction

app = Flask(__name__)

cnn_model = tf.keras.models.load_model('models/cnn_model.h5')
dp_model = tf.keras.models.load_model('models/dp_model.h5')
knn_model = joblib.load('models/knn_model.joblib')
random_forest_model = joblib.load('models/random_forest_model.joblib')
xgb_model = Booster()
xgb_model.load_model('models/xgb_model.json')
dp_ensemble = []
for i in range(1, 11):
    dp_ensemble.append(
      tf.keras.models.load_model(f'models/dp_ensemble/dp_model_{i}.h5')
    )

def predict_audio(predict_func, request):
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        audio_data, sample_rate = librosa.load(file, sr=None)
        features = extract_features(audio_data, sample_rate)
        prepared_features = prepare_features_for_model(features)
        probabilities_list = predict_func(prepared_features)
        return jsonify(probabilities_list)

    return jsonify({'error': 'Invalid file'}), 400

@app.route('/predict/xgb', methods=['POST'])
def xgb_predict():
    return predict_audio(
        partial(get_xgb_prediction, xgb_model),
        request
    )


@app.route('/predict/cnn', methods=['POST'])
def cnn_predict():
    return predict_audio(
        partial(get_cnn_prediction, cnn_model),
        request
    )

@app.route('/predict/random_forest', methods=['POST'])
def random_forest_predict():
    return predict_audio(
        partial(get_random_forest_prediction, random_forest_model),
        request
    )

@app.route('/predict/knn', methods=['POST'])
def knn_predict():
    return predict_audio(
        partial(get_knn_prediction, knn_model),
        request
    )

@app.route('/predict/deep_learning', methods=['POST'])
def dp_predict():
    return predict_audio(
        partial(get_dp_prediction, dp_model),
        request
    )

@app.route('/predict/deep_learning/ensemble', methods=['POST'])
def dp_ensemble_predict():
    return predict_audio(
        partial(get_dp_ensemble_prediction, dp_ensemble),
        request
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)