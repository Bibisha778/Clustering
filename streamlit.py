import streamlit as st
import pandas as pd
import pickle
from src.data.make_dataset import load_and_preprocess_data
from src.features.build_features import build_features
from src.models.train_model import train_kmeans
from src.models.predict_model import predict_cluster
from src.models.visualization import plot_clusters

st.set_page_config(page_title="🛍️ Customer Clustering", layout="centered")
st.title("🛍️ Customer Segmentation Using KMeans")

model, scaler = None, None

if st.sidebar.button("Train Clustering Model"):
    df = load_and_preprocess_data("final.csv")
    X = build_features(df)
    model, scaler = train_kmeans(X)
    labels = model.predict(scaler.transform(X))
    st.success("Model trained and saved.")
    st.write("### Cluster Visualization")
    plot_clusters(scaler.transform(X), labels)

try:
    with open("models/kmeans.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    st.warning("Please train the model first using the sidebar.")

st.header("📋 Enter Customer Info")
with st.form("customer_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 10, 100, 30)
    income = st.slider("Annual Income (k$)", 10, 200, 50)
    score = st.slider("Spending Score (1-100)", 1, 100, 50)
    submit = st.form_submit_button("Predict Segment")

if submit and model and scaler:
    gender_num = 1 if gender == "Male" else 0
    input_df = pd.DataFrame([{
        "Gender": gender_num,
        "Age": age,
        "Annual_Income": income,
        "Spending_Score": score
    }])
    cluster = predict_cluster(model, scaler, input_df)[0]
    st.subheader("🧠 Predicted Customer Segment")
    st.success(f"This customer belongs to segment #{cluster}")