from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data(data, test_size=0.2, random_state=42):
    X = data.drop(['filename', 'label'], axis=1)
    y = data['label']
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    X = StandardScaler().fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    return X_train, X_val, y_train, y_val, encoder
