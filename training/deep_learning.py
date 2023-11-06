import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report


# Function to create a base model
def create_base_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics='accuracy')
    return model


# Function to create a model with dropout
def create_model_with_dropout(input_shape, dropout):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics='accuracy')
    return model


# Function to train a model
def train_model(model, X_train, y_train, X_val, y_val, epochs=150, batch_size=32):
    history = model.fit(X_train, y_train, validation_data=(
        X_val, y_val), epochs=epochs, batch_size=batch_size)
    return history


# Function to evaluate a model
def evaluate_model(model, X_val, y_val, encoder):
    y_val_pred = np.argmax(model.predict(X_val), axis=1)
    report = classification_report(
        y_true=y_val,
        y_pred=y_val_pred,
        target_names=encoder.classes_
    )
    return report


# Function to create and train multiple models for an ensemble
def train_ensemble_models(X_train, y_train, num_models=10, epochs=150, batch_size=32):
    models = []
    for i in range(num_models):
        print(f"\n\n### Training model {i+1}\n")
        model = create_model_with_dropout(
            input_shape=(X_train.shape[1],), dropout=0.2)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        models.append(model)
    return models


# Function to combine predictions from multiple models
def ensemble_predictions(models, X_val):
    predictions = [model.predict(X_val) for model in models]
    mean_predictions = np.mean(predictions, axis=0)
    ensemble_pred = np.argmax(mean_predictions, axis=1)
    return ensemble_pred


# Function to evaluate the ensemble of models
def evaluate_ensemble(models, X_val, y_val, encoder):
    ensemble_pred = ensemble_predictions(models, X_val)
    report = classification_report(
        y_val, ensemble_pred, target_names=encoder.classes_)
    return report
