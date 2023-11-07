from functools import partial

from flask import Flask, request, jsonify
import librosa
import joblib
import tensorflow as tf
from xgboost import Booster
import pandas as pd

from prediction.extract import extract_features
from prediction.xgboost import get_xgb_prediction
from prediction.CNN import get_cnn_prediction
from prediction.random_forest import get_random_forest_prediction
from prediction.KNN import get_knn_prediction
from prediction.deep_learning import (
  get_dp_prediction,
  get_dp_ensemble_prediction
)

app = Flask(__name__)

# Define the CNN model
cnn_model = tf.keras.models.load_model('models/cnn_model.h5')

# Define the KNN model
knn_model = joblib.load('models/knn_model.joblib')

# Define the Random Forest model
random_forest_model = joblib.load('models/random_forest_model.joblib')

# Define the XGBoost model
xgb_model = Booster()
xgb_model.load_model('models/xgb_model.json')

# Define deep learning model and an ensemble of deep learning models
dp_model = tf.keras.models.load_model('models/dp_model.h5')
dp_ensemble = []
for i in range(1, 11):
    dp_ensemble.append(
      tf.keras.models.load_model(f'models/dp_ensemble/dp_model_{i}.h5')
    )

# Define the feature columns
feature_columns = pd.read_csv('Data/features_3_sec.csv',nrows=0).columns
feature_columns = feature_columns.drop(['filename', 'label'])


def predict_audio(predict_func, request):
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file:
        return jsonify({'error': 'Invalid file'}), 400

    audio_data, sample_rate = librosa.load(file, sr=None)
    features = extract_features(audio_data, sample_rate, feature_columns)
    probabilities_list = predict_func(features)

    return jsonify(probabilities_list)


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


@app.route('/predict/random-forest', methods=['POST'])
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


@app.route('/predict/deep-learning', methods=['POST'])
def dp_predict():
    return predict_audio(
        partial(get_dp_prediction, dp_model),
        request
    )


@app.route('/predict/deep-learning/ensemble', methods=['POST'])
def dp_ensemble_predict():
    return predict_audio(
        partial(get_dp_ensemble_prediction, dp_ensemble),
        request
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)