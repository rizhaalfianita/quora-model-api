import numpy as np

class MultinomialNaiveBayes:

    def __init__(self, alpha=1): # 
        self.alpha = alpha  # Parameter untuk laplacian smoothing
        self.class_probs = {}  # Probabilitas prior
        self.feature_probs = {}  # Probabilitas likelihood

    def fit(self, X, y):
        # X: Matriks TF-IDF, y: Label kelas
        num_docs, num_features = X.shape
        unique_classes = np.unique(y)

        # Perhitungan probabilitas prior
        for label in unique_classes:
            self.class_probs[label] = np.sum(y == label) / num_docs

        # Perhitungan probabilitas likelihood
        for label in unique_classes:
            class_docs = X[y == label]
            total_word_count = np.sum(class_docs)
            self.feature_probs[label] = (np.sum(class_docs, axis=0) + self.alpha) / (total_word_count + self.alpha * num_features)

    def predict(self, X):
        # X: Matriks TF-IDF uji
        predictions = []
        for doc in X:
            posterior_probs = {}
            for label, class_prob in self.class_probs.items():
                # Perhitungan probabilitas posterior tanpa normalisasi
                posterior_prob = np.log(class_prob) + np.sum(np.log(self.feature_probs[label]) * doc)
                posterior_probs[label] = posterior_prob

            # Pilih kelas dengan probabilitas posterior tertinggi
            predicted_label = max(posterior_probs, key=posterior_probs.get)
            predictions.append(predicted_label)

        return predictions
