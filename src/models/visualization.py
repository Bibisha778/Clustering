import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

def plot_clusters(X_scaled, labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_vis = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
    df_vis["Cluster"] = labels
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df_vis, x="PCA1", y="PCA2", hue="Cluster", palette="tab10")
    plt.title("Customer Segments")
    st.pyplot(plt.gcf())