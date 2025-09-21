import numpy as np
import joblib
from sklearn.calibration import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

class FaceRecognizer:
    def __init__(self, embedder, n_neighbors=3):
        self.embedder = embedder
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.X_emb = []
        self.y = []
        self.le = LabelEncoder()

    def get_embeddings(self, X, y):
        X_emb, y_emb = [], []
        for img, label in zip(X, y):
            print(img.shape)
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            print(img.shape)
            results = self.embedder.extract(img, threshold=0.95)
            if len(results) == 0:
                continue
            emb = results[0]["embedding"]
            X_emb.append(emb)
            y_emb.append(label)
        return np.array(X_emb), np.array(y_emb)

    def fit(self, X, y):
        y_encoded = self.le.fit_transform(y)
        X_emb, y_emb = self.get_embeddings(X, y_encoded)
        self.X_emb.extend(X_emb)
        self.y.extend(y_emb)
        self.knn.fit(self.X_emb, self.y)

    def add_person(self, X_new, y_new):
        self.fit(X_new, y_new)
        return True

    def predict(self, X, y=None):
        X_emb, _ = self.get_embeddings(X, y if y is not None else [0] * len(X))
        if len(X_emb) == 0:
            return [], []
        pred_indices = self.knn.predict(X_emb)
        pred_proba = self.knn.predict_proba(X_emb)
       
        confidences = pred_proba.max(axis=1)
        pred_names = self.le.inverse_transform(pred_indices)
        return pred_names, confidences

    def save(self, path="models"):
        joblib.dump((self.X_emb, self.y), f"{path}/face_data.pkl")
        joblib.dump(self.le, f"{path}/label_encoder.pkl")

    def load(self, path="models"):
        self.X_emb, self.y = joblib.load(f"{path}/face_data.pkl")
        self.knn.fit(self.X_emb, self.y)
        self.le = joblib.load(f"{path}/label_encoder.pkl")