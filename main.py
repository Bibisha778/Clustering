from src.data.make_dataset import load_and_preprocess_data
from src.features.build_features import build_features
from src.models.train_model import train_kmeans
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))


if __name__ == "__main__":
    df = load_and_preprocess_data("final.csv")
    X = build_features(df)
    model, scaler = train_kmeans(X)
    print("Model trained and saved.")