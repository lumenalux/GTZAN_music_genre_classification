# MLZoomcamp Midterm project: Music Genre Classification

## Overview

This project aims to classify music genres using various machine learning models. The project utilizes the GTZAN dataset, which is the most-used public dataset for evaluation in music genre recognition (MGR) research. The data includes audio files across 10 different genres, each 30 seconds long, and their corresponding Mel Spectrograms for visual representation

## Problem Statement

The challenge is to accurately classify the genre of a given piece of music. This involves understanding the audio file, visualizing it, extracting features, and then using these features to predict the genre with machine learning models.

## Installation

### Using a Virtual Environment

1. Clone the repository:

```sh
git clone https://github.com/lumenalux/GTZAN_music_genre_classification
cd GTZAN_music_genre_classification
```

1.  Create and activate a virtual environment:

- On macOS and Linux:

`python3 -m venv venv
source venv/bin/activate`

- On Windows:

`python -m venv venv
.\venv\Scripts\activate`

1.  Install the dependencies:

`pip install -r requirements.txt`

### Using Docker

1.  Build the Docker image:

`docker build -t midterm .`

1.  Run the Docker container:

`docker run -p 8080:5000 midterm`

### Using Kubernetes

1.  Apply the Kubernetes manifests to create the deployment and service:

`kubectl apply -f deployment.yaml
kubectl apply -f service.yaml`

1.  Get the external IP of your service:

`kubectl get services`

## Usage

After installing and starting the service, you can use the following endpoints to classify music genres:

- XGBoost Classifier

  `POST /predict/xgb`

  Use this endpoint to classify music using the XGBoost model.

- CNN Classifier

  `POST /predict/cnn`

  Use this endpoint to classify music using the Convolutional Neural Network model.

- Random Forest Classifier

  `POST /predict/random_forest`

  Use this endpoint to classify music using the Random Forest model.

- KNN Classifier

  `POST /predict/knn`

  Use this endpoint to classify music using the K-Nearest Neighbors model.

- Deep Learning Classifier

  `POST /predict/deep_learning`

  Use this endpoint to classify music using a Deep Learning model.

- Deep Learning Ensemble Classifier

  `POST /predict/deep_learning/ensemble`

  Use this endpoint to classify music using an ensemble of Deep Learning models.

To make a prediction, send a POST request to the desired endpoint with a form-data body containing the audio file under the key "file". For example:

shCopy code

`curl -X POST -F "file=@/path-to-your-file/music.wav" http://localhost:8080/predict/xgb`

## EDA Notebook

An exploratory data analysis (EDA) for this project is provided in the `notebooks/EDA.ipynb` notebook. It contains visualizations and insights derived from the dataset.

## Testing

You can test the endpoints locally after running the application via Docker or directly through Python, or you can use the external IP provided by the Kubernetes service to test the deployed application.
