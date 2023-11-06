import os
import pandas as pd
import joblib
import numpy as np

from tensorflow import keras
from keras.utils import to_categorical

from training.deep_learning import train_ensemble_models
from training.preparation import preprocess_data
from training.CNN import train_cnn_model, create_cnn_model
from training.random_forest import train_random_forest, experiment_with_estimators as rf_experiment
from training.KNN import train_knn, knn_experiments
from training.XGBoost import train_xgboost, xgboost_experiments

# Load data
data_path = 'Data/features_3_sec.csv'
data = pd.read_csv(data_path)

# Prepare the data (assuming you have a function for that)
X_train, X_val, y_train, y_val, encoder = preprocess_data(data)

# Define a directory to save the models
model_dir = 'saved_models'
os.makedirs(model_dir, exist_ok=True)

# Data preparation for CNN model
X_val_reshaped = np.array(X_val).reshape(X_val.shape[0], X_val.shape[1], 1)
y_val_encoded = encoder.fit_transform(y_val)
y_val_categorical = to_categorical(y_val_encoded)

y_train_encoded = encoder.fit_transform(y_train)
y_train_categorical = to_categorical(y_train_encoded)

# Train CNN model
cnn_input_shape = (X_val_reshaped.shape[1], 1)
cnn_model = create_cnn_model(cnn_input_shape, len(np.unique(y_val)))
cnn_model = train_cnn_model(cnn_model,
                            X_train,
                            y_train_categorical,
                            X_val_reshaped,
                            y_val_categorical)
cnn_model.save(os.path.join(model_dir, 'cnn_model.h5'))

# Train and experiment with Random Forest models
random_forest_model = train_random_forest(X_train, y_train)
joblib.dump(
    random_forest_model,
    os.path.join(model_dir, 'random_forest_model.joblib')
)
# rf_experiment(X_train, y_train, X_val, y_val, encoder, [5, 10, 25, 50, 100, 200, 500])

# Train and experiment with KNN models
knn_model = train_knn(X_train, y_train)
joblib.dump(knn_model, os.path.join(model_dir, 'knn_model.joblib'))
# knn_experiments(X_train, y_train, X_val, y_val, encoder, [2, 3, 4, 5, 10, 15])

# Train and experiment with XGBoost models
xgb_model = train_xgboost(X_train, y_train)
xgb_model.save_model(os.path.join(model_dir, 'xgb_model.json'))
# xgboost_experiments(X_train, y_train, X_val, y_val, encoder, [100, 200, 500])


# Train and evaluate deep learning ensemble of models
num_models = 10
ensemble_models = train_ensemble_models(
    X_train, y_train, num_models=num_models, epochs=150, batch_size=32)

# Save each model in the ensemble
for idx, model in enumerate(ensemble_models):
    model.save(
        os.path.join(
            model_dir,
            f'dp_ensemble/dp_model_{idx+1}.h5'
        )
    )
