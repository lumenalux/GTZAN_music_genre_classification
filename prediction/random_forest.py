def get_random_forest_prediction(model, features):
        prediction = [0.0] * 10
        prediction[model.predict(features).tolist()[0] - 1] = 1.0
        return prediction