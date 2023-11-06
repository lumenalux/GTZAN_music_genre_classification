from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def preprocess_data(data, features, target):
    # Splitting data into features and labels
    X = data[features]
    y = data[target]

    # Encoding labels and feature normalization
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    X_scaled = StandardScaler().fit_transform(X)

    return X_scaled, y_encoded, encoder


def split_data(X, y, test_size=0.2, random_state=42):
    # Split data into training and validation parts
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_val, y_val, encoder):
    # Estimate prediction
    y_val_predict = model.predict(X_val)

    # Evaluation
    report = classification_report(
        y_val, y_val_predict, target_names=encoder.classes_)
    return report


def experiment_with_estimators(X_train, y_train, X_val, y_val, encoder, estimators_list):
    for n_estimators in estimators_list:
        print(f"\n\n# n_estimators: {n_estimators}\n")
        classifier = train_random_forest(X_train, y_train, n_estimators)
        report = evaluate_model(classifier, X_val, y_val, encoder)
        print(report)
