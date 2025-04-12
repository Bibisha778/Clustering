from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle

def train_kmeans(X, n_clusters=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)

    with open("models/kmeans.pkl", "wb") as f:
        pickle.dump(kmeans, f)

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return kmeans, scaler