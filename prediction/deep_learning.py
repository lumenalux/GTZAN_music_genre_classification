import numpy as np

def get_dp_prediction(model, features):
        return np.argmax(model.predict(features), axis=1)


def get_dp_ensemble_prediction(models, features):
        predictions = [model.predict(features) for model in models]
        mean_predictions = np.mean(predictions, axis=0)
        ensemble_predictions = np.argmax(mean_predictions, axis=1)

        return ensemble_predictions