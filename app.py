import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Page settings
st.set_page_config(page_title="K-Means Clustering App with Iris", layout="wide")

# App title
st.markdown("<h1 style='text-align: center;'>🔍 K-Means Clustering App with Iris Dataset</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Configure Clustering")
k = st.sidebar.slider("Select number of clusters (K)", min_value=2, max_value=10, value=3)

# Load Iris dataset
iris = load_iris()
X = iris.data
feature_names = iris.feature_names

# K-Means model
model = KMeans(n_clusters=k, random_state=42)
y_kmeans = model.fit_predict(X)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
centers_pca = pca.transform(model.cluster_centers_)

# Plotting
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='tab10', s=50)



# Labels & title
ax.set_title("Clusters (2D PCA Projection)")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")

# Add legend with cluster labels
handles, labels = scatter.legend_elements()
labels = [f"Cluster {i}" for i in range(len(handles))]
ax.legend(handles, labels, title="Clusters")

# Show plot in Streamlit
st.pyplot(fig)
