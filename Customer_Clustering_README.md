# 🛍️ Customer Segmentation Using KMeans

This project segments mall customers into behavioral groups using unsupervised KMeans clustering. The app also visualizes the clusters using PCA.

## 🚀 Features
- Clusters customers into 5 segments based on age, income, spending score, etc.
- Preprocessing includes encoding gender and scaling features
- PCA visualization of clusters
- Streamlit UI for entering new customer info and predicting their segment

## 📂 Project Structure
- `streamlit.py`: Web app
- `main.py`: Trainer
- `models/`: Stores KMeans and scaler
- `src/`: Data loading, feature building, clustering, and visualization

## ▶️ Run Locally
```bash
streamlit run streamlit.py
```

## 🧰 Requirements
- streamlit
- pandas
- scikit-learn
- matplotlib
- seaborn