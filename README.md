# MLZoomcamp Midterm project: Music Genre Classification

## Overview

This project applies machine learning techniques to automatically classify music into genres based on audio data. The models are trained and evaluated using the GTZAN dataset, a popular benchmark dataset for music genre recognition. The goal is to accurately predict the genre of a music clip given its audio waveform and spectrogram image.

## Problem Statement

This project aims to explore machine learning architectures for music genre recognition using the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). A key focus is understanding how different network architectures impact performance on this multi-class classification task. The project only uses features from the [Data/features_3_sec.csv](Data/features_3_sec.csv) file. But you can download the entire dataset to try how different models work on music files from GTZAN dataset:

```sh
kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification
unzip gtzan-dataset-music-genre-classification.zip
```

## Installation

### Using a Virtual Environment

1. Clone the repository:

```sh
git clone https://github.com/lumenalux/GTZAN_music_genre_classification
cd GTZAN_music_genre_classification
```

1.  Create and activate a virtual environment:

- On macOS and Linux:

```sh
python3 -m venv venv
source venv/bin/activate
```

- On Windows:

```sh
python -m venv venv
.\venv\Scripts\activate
```

1.  Install the dependencies:

```sh
pip install -r requirements.txt
```

### Using Docker

1.  Build the Docker image:

```sh
docker build -t midterm .
```

1.  Run the Docker container:

```sh
docker run -d -p 5000:5000 midterm
```

### Using Kubernetes

1.  Apply the Kubernetes manifests to create the deployment and service:

```sh
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

1.  Get the external IP of your service:

```sh
kubectl get services
```

## Usage

After installing and starting the service, you can use the following endpoints to classify music genres:

- XGBoost Classifier

  `POST /predict/xgb`

  Use this endpoint to classify music using the XGBoost model.

- CNN Classifier

  `POST /predict/cnn`

  Use this endpoint to classify music using the Convolutional Neural Network model.

- Random Forest Classifier

  `POST /predict/random-forest`

  Use this endpoint to classify music using the Random Forest model.

- KNN Classifier

  `POST /predict/knn`

  Use this endpoint to classify music using the K-Nearest Neighbors model.

- Deep Learning Classifier

  `POST /predict/deep-learning`

  Use this endpoint to classify music using a Deep Learning model.

- Deep Learning Ensemble Classifier

  `POST /predict/deep-learning/ensemble`

  Use this endpoint to classify music using an ensemble of Deep Learning models.

To make a prediction, send a POST request to the desired endpoint with a form-data body containing the audio file under the key "file". For example:

```sh
curl -X POST -F "file=@/path-to-your-file/music.wav" http://localhost:5000/predict/xgb
```

## Using Locally with [Minikube](https://minikube.sigs.k8s.io/docs/)

Once deployed locally using Kubernetes and Minikube, get the URL for the `midterm-service` using:

```sh
minikube service midterm-service
```

This will output something like:

```
|-----------|-----------------|-------------|---------------------------|
| NAMESPACE |      NAME       | TARGET PORT |            URL            |
|-----------|-----------------|-------------|---------------------------|
| default   | midterm-service |        5000 | http://192.168.49.2:32150 |
|-----------|-----------------|-------------|---------------------------|
```

Make a note of the `URL` (e.g. `http://192.168.49.2:32150`).

You can now make predictions by sending a POST request with an audio file:

```sh
curl -X POST\
  -F "file=@/path/to/example.wav"\
  http://192.168.49.2:32150/predict/deep-learn
```

The API will return a JSON response with the prediction:

```json
{ "file name": "example.wav", "label": "pop" }
```

## Exploratory Data Analysis (EDA)

An exploratory data analysis (EDA) for this project is provided in the [notebooks/EDA.ipynb](notebooks/EDA.ipynb) notebook. It contains visualizations and insights derived from the dataset.

## ML models used in this project

### K-Nearest Neighbors (KNN)

The [notebooks/KNN.ipynb](notebooks/KNN.ipynb) notebook explores the use of K-Nearest Neighbors for music genre classification. KNN classifies songs based on feature similarity, using the `features_3_sec.csv` data to find the closest genre match among known samples.

Endpoint:

`POST /predict/knn`

Performance:

```
              precision    recall  f1-score   support

       blues       0.89      0.92      0.91       208
   classical       0.91      0.97      0.94       203
     country       0.82      0.85      0.84       186
       disco       0.89      0.93      0.91       199
      hiphop       0.93      0.90      0.92       218
        jazz       0.92      0.89      0.90       192
       metal       0.99      0.98      0.98       204
         pop       0.94      0.91      0.92       180
      reggae       0.93      0.94      0.93       211
        rock       0.92      0.85      0.88       197

    accuracy                           0.91      1998
   macro avg       0.91      0.91      0.91      1998
weighted avg       0.92      0.91      0.91      1998
```

### Random Forest

The [notebooks/RandomForest.ipynb](notebooks/RandomForest.ipynb) notebook investigates the Random Forest algorithm's effectiveness in classifying music genres

Endpoint:

`POST /predict/random-forest`

Performance:

```
              precision    recall  f1-score   support

       blues       0.91      0.87      0.89       208
   classical       0.93      0.98      0.95       203
     country       0.78      0.83      0.80       186
       disco       0.87      0.86      0.87       199
      hiphop       0.94      0.89      0.91       218
        jazz       0.86      0.92      0.89       192
       metal       0.89      0.96      0.92       204
         pop       0.92      0.93      0.93       180
      reggae       0.90      0.88      0.89       211
        rock       0.88      0.75      0.81       197

    accuracy                           0.89      1998
   macro avg       0.89      0.89      0.89      1998
weighted avg       0.89      0.89      0.89      1998
```

### Convolutional Neural Network (CNN)

The [notebooks/simpleCNN.ipynb](notebooks/simpleCNN.ipynb) notebook demonstrates the application of Convolutional Neural Networks to classify music genres. CNN model utilizes features from `features_3_sec.csv` to recognize distinct genre characteristics.

Endpoint:

`POST /predict/cnn`

Classification Report:

```
              precision    recall  f1-score   support

           0       0.90      0.90      0.90       208
           1       0.92      0.94      0.93       203
           2       0.84      0.85      0.84       186
           3       0.85      0.87      0.86       199
           4       0.91      0.93      0.92       218
           5       0.86      0.93      0.89       192
           6       0.89      0.99      0.93       204
           7       0.95      0.92      0.94       180
           8       0.89      0.88      0.89       211
           9       0.91      0.70      0.79       197

    accuracy                           0.89      1998
   macro avg       0.89      0.89      0.89      1998
weighted avg       0.89      0.89      0.89      1998
```

### XGBoost

The [notebooks/XGBoost.ipynb](notebooks/XGBoost.ipynb) notebook highlights the application of XGBoost, a powerful gradient boosting framework, to our classification task. Trained on `features_3_sec.csv`, the model leverages sequential improvements to achieve precise genre predictions.

Endpoint:

`POST /predict/xgb`

Performance:

```
              precision    recall  f1-score   support

       blues       0.91      0.89      0.90       208
   classical       0.93      0.97      0.95       203
     country       0.83      0.90      0.86       186
       disco       0.91      0.88      0.90       199
      hiphop       0.96      0.92      0.94       218
        jazz       0.88      0.91      0.89       192
       metal       0.95      0.96      0.95       204
         pop       0.94      0.96      0.95       180
      reggae       0.95      0.92      0.94       211
        rock       0.92      0.86      0.89       197

    accuracy                           0.92      1998
   macro avg       0.92      0.92      0.92      1998
weighted avg       0.92      0.92      0.92      1998
```

### Deep Learning

The [notebooks/DeepLearning.ipynb](notebooks/DeepLearning.ipynb) notebook shows how deep neural networks can identify and classify music genres

Endpoint:

`POST /predict/deep-learning`

Performance:

```
              precision    recall  f1-score   support

       blues       0.94      0.94      0.94       208
   classical       0.93      0.98      0.96       203
     country       0.88      0.90      0.89       186
       disco       0.91      0.94      0.93       199
      hiphop       0.99      0.92      0.95       218
        jazz       0.94      0.91      0.92       192
       metal       0.96      0.99      0.98       204
         pop       0.95      0.94      0.95       180
      reggae       0.94      0.95      0.95       211
        rock       0.91      0.88      0.90       197

    accuracy                           0.94      1998
   macro avg       0.94      0.94      0.94      1998
weighted avg       0.94      0.94      0.94      1998
```

### Deep Learning Ensemble

The [notebooks/DeepLearningEnsemble.ipynb](notebooks/DeepLearningEnsemble.ipynb) notebook presents the strength of combining multiple deep learning models to enhance prediction accuracy

Endpoint:

`POST /predict/deep-learning/ensemble`

Performance:

```
              precision    recall  f1-score   support

       blues       0.98      0.95      0.96       208
   classical       0.94      0.98      0.96       203
     country       0.93      0.95      0.94       186
       disco       0.96      0.92      0.94       199
      hiphop       0.97      0.95      0.96       218
        jazz       0.94      0.98      0.96       192
       metal       0.98      0.99      0.98       204
         pop       0.96      0.96      0.96       180
      reggae       0.96      0.97      0.96       211
        rock       0.95      0.91      0.93       197

    accuracy                           0.96      1998
   macro avg       0.96      0.96      0.96      1998
weighted avg       0.96      0.96      0.96      1998
```
