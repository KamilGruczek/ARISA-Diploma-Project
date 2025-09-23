import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
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
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            results = self.embedder.extract(img, threshold=0.95)
            if len(results) == 0:
                continue
            emb = results[0]["embedding"]
            X_emb.append(emb)
            y_emb.append(label)
        return np.array(X_emb), np.array(y_emb)

    def fit(self):
        if not self.X_emb or not self.y:
            return
        self.le.fit(self.y)
        y_encoded = self.le.transform(self.y)
        self.knn.fit(self.X_emb, y_encoded)

    def add_person(self, X_new, y_new):
        # Walidacja: tylko stringi, nie liczby!
        for label in y_new:
            if not isinstance(label, str) or label.isdigit():
                print(f"Nieprawidłowa etykieta: {label}")
                return False
        X_emb, y_emb = self.get_embeddings(X_new, y_new)
        if len(X_emb) == 0:
            return False
        print("Adding person(s):", y_emb)
        self.X_emb.extend(X_emb)
        self.y.extend(y_emb)
        self.fit()
        return True

    def predict(self, X):
        if not self.X_emb or not self.y:
            return [], []
        pred_names = []
        confidences = []
        emb_idx = 0
        for img in X:
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            results = self.embedder.extract(img, threshold=0.95)
            if len(results) == 0:
                pred_names.append("Unknown")
                confidences.append(0.0)
            else:
                emb = results[0]["embedding"]
                pred_index = self.knn.predict([emb])[0]
                pred_proba = self.knn.predict_proba([emb])[0]
                confidence = pred_proba.max()
                try:
                    name = self.le.inverse_transform([pred_index])[0]
                except Exception:
                    name = "Unknown"
                pred_names.append(name)
                confidences.append(confidence)
                emb_idx += 1
        return np.array(pred_names), np.array(confidences)

    def save(self, path="models"):
        joblib.dump((self.X_emb, self.y), f"{path}/face_data.pkl")
        joblib.dump(self.le, f"{path}/label_encoder.pkl")

    def load(self, path="models"):
        self.X_emb, self.y = joblib.load(f"{path}/face_data.pkl")
        self.le = joblib.load(f"{path}/label_encoder.pkl")
        self.fit()
