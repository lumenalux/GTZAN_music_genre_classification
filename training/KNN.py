# KNN.py

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import pandas as pd


def preprocess_data(data, feature_cols, label_col):
    # Splitting data into features and labels
    X = data[feature_cols]
    y = data[label_col]

    # Encoding labels and feature normalization
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    X_scaled = StandardScaler().fit_transform(X)

    return X_scaled, y_encoded, encoder


def split_data(X, y, test_size=0.2, random_state=42):
    # Split data into training and validation parts
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_knn(X_train, y_train, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn


def evaluate_knn(knn, X_val, y_val, encoder):
    # Estimate prediction
    y_knn_predict = knn.predict(X_val)

    # Evaluation
    report = classification_report(
        y_val, y_knn_predict, target_names=encoder.classes_)
    return report


def knn_experiments(X_train, y_train, X_val, y_val, encoder, neighbors_list):
    for n_neighbors in neighbors_list:
        print(f"\n\n# n_neighbors: {n_neighbors}\n")
        knn = train_knn(X_train, y_train, n_neighbors)
        report = evaluate_knn(knn, X_val, y_val, encoder)
        print(report)
