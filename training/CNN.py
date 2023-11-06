import numpy as np
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# Function to create the CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(filters=64, kernel_size=5,
               activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Function to train the CNN model
def train_cnn_model(model, X_train, y_train, X_val, y_val, epochs=15, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs,
              batch_size=batch_size, validation_data=(X_val, y_val))
    return model


# Function to evaluate the CNN model
def evaluate_cnn_model(model, X_val, y_val, encoder):
    y_val_pred = np.argmax(model.predict(X_val), axis=1)
    y_val_encoded = encoder.transform(y_val)
    report = classification_report(
        y_true=y_val_encoded,
        y_pred=y_val_pred,
        target_names=encoder.classes_
    )
    return report


# Function to create a feature extractor from the CNN
def create_feature_extractor(input_shape, num_classes):
    feature_extractor = Sequential([
        Conv1D(filters=64, kernel_size=5,
               activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    feature_extractor.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return feature_extractor


# Function to train and evaluate a Random Forest classifier using features extracted by the CNN
def train_evaluate_random_forest(feature_extractor, X_train, y_train, X_val, y_val):
    features = feature_extractor.predict(X_train)
    random_forest_model = RandomForestClassifier(n_estimators=5)
    random_forest_model.fit(features, y_train)

    val_features = feature_extractor.predict(X_val)
    predictions = random_forest_model.predict(val_features)

    report = classification_report(
        y_val, predictions, target_names=LabelEncoder().fit(y_val).classes_.astype(str))
    return report


# Function to train and evaluate a KNN classifier using features extracted by the CNN
def train_evaluate_knn(feature_extractor, X_train, y_train, X_val, y_val, n_neighbors=8):
    features = feature_extractor.predict(X_train)
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(features, y_train)

    val_features = feature_extractor.predict(X_val)
    predictions = knn_model.predict(val_features)

    report = classification_report(
        y_val, predictions, target_names=LabelEncoder().fit(y_val).classes_.astype(str))
    return report
