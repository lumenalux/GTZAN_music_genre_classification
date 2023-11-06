from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def train_knn(X_train, y_train, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn


def evaluate_knn(knn, X_val, y_val, encoder):
    # Estimate prediction
    y_knn_predict = knn.predict(X_val)

    # Evaluation
    report = classification_report(
        y_true=y_val,
        y_pred=y_knn_predict,
        target_names=encoder.classes_
    )
    return report


def knn_experiments(X_train, y_train, X_val, y_val, encoder, neighbors_list):
    for n_neighbors in neighbors_list:
        print(f"\n\n# n_neighbors: {n_neighbors}\n")
        knn = train_knn(X_train, y_train, n_neighbors)
        report = evaluate_knn(knn, X_val, y_val, encoder)
        print(report)
