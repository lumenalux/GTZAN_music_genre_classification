# MLZoomcamp Midterm project: Music Genre Classification

## Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Installation](#installation)
  - [Using a Virtual Environment](#using-a-virtual-environment)
  - [Using Docker](#using-docker)
  - [Using Kubernetes](#using-kubernetes)
- [Usage](#usage)
- [Using Locally with Minikube](#using-locally-with-minikube)
- [Kubernetes Test](#kubernetes-test)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [ML models used in this project](#ml-models-used-in-this-project)
  - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
  - [Random Forest](#random-forest)
  - [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
  - [XGBoost](#xgboost)
  - [Deep Learning](#deep-learning)
  - [Deep Learning Ensemble](#deep-learning-ensemble)

## Overview

This project applies machine learning techniques to automatically classify music into genres based on audio data. The models are trained and evaluated using the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), a popular benchmark dataset for music genre recognition. The goal is to accurately predict the genre of a music clip given its audio waveform and spectrogram image.

## Problem Statement

This project aims to explore machine learning architectures for music genre recognition using the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). A key focus is understanding how different network architectures impact performance on this multi-class classification task. The project only uses features from the [Data/features_3_sec.csv](Data/features_3_sec.csv) file. But you can download the entire dataset to try how different models work on music files from GTZAN dataset:

```sh
kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification
unzip gtzan-dataset-music-genre-classification.zip
```

The model's versatility allows it to recognize a spectrum of ten distinct genres: `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`

The practical application of this project extends to various domains including digital music services, content creators, and music enthusiasts. By automating the process of genre classification, the project provides a means for:

1.  Music streaming platforms to organize vast libraries of songs, enhancing user experience through better music recommendation systems and more accurate genre-based sorting.
2.  Artists and producers to classify their music without subjective bias, ensuring that their work reaches the intended audience through appropriate channels.
3.  Researchers and hobbyists in the field of musicology to analyze trends and patterns in music genres, aiding in academic studies and personal projects.

This machine learning-based approach simplifies the task of categorizing music by genre, can be highly subjective. The tool is simple to use; one can classify a music track by sending an audio file through a REST API endpoint, receiving the genre prediction in return:

```sh
curl -X POST \
  -F "file=@/path/to/your/song.wav" \
  http://yourserveraddress:port/predict/deep-learning
```

Upon submission, the user receives a JSON response containing the file name and the predicted genre label:

```json
{
  "file name": "song.wav",
  "label": "genre"
}
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
  http://192.168.49.2:32150/predict/deep-learning
```

The API will return a JSON response with the prediction:

```json
{ "file name": "example.wav", "label": "pop" }
```

## Kubernetes Test

![Kubernetes start](docs/kubernetes-start.png)
![Kubernetes info](docs/kubernetes-info.png)
![Minikube service](docs/minikube-service.png)
![Usage with kubernetes](docs/usage-kubernetes.png)

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

The [notebooks/random_forest.ipynb](notebooks/random_forest.ipynb) notebook investigates the Random Forest algorithm's effectiveness in classifying music genres

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

The [notebooks/CNN.ipynb](notebooks/CNN.ipynb) notebook demonstrates the application of Convolutional Neural Networks to classify music genres. CNN model utilizes features from `features_3_sec.csv` to recognize distinct genre characteristics.

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

The [notebooks/deep_learning.ipynb](notebooks/deep_learning.ipynb) notebook shows how deep neural networks can identify and classify music genres

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

The [notebooks/deep_learning_ensemble.ipynb](notebooks/deep_learning_ensemble.ipynb) notebook presents the strength of combining multiple deep learning models to enhance prediction accuracy

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
