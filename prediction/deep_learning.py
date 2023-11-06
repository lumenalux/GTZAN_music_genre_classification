import numpy as np

def get_dp_prediction(model, features):
        return model.predict(features).tolist()[0]


def get_dp_ensemble_prediction(models, features):
        predictions = [model.predict(features) for model in models]
        mean_predictions = np.mean(predictions, axis=0)
        ensemble_predictions = np.argmax(mean_predictions, axis=1)

        prediction = [0.0] * 10
        prediction[ensemble_predictions.tolist()[0] - 1] = 1.0
        return prediction