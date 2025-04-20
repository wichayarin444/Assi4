# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 10:00:00 2024  # Corrected date
@author: Streamlit User Example # Corrected author/context
"""

# app.py
import streamlit as st
import numpy as np # Added for array manipulation
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris # To load the Iris dataset as implied by image title
from sklearn.preprocessing import StandardScaler # To scale data (good practice for K-Means)
from sklearn.decomposition import PCA # To perform PCA for 2D visualization
from sklearn.cluster import KMeans # To perform K-Means clustering

# --- Page Configuration ---
st.set_page_config(page_title="K-Means Clustering App", layout="wide") # Use wide layout like image
st.set_option('deprecation.showPyplotGlobalUse', False) # Suppress warning for st.pyplot()

# --- Data Loading and Preparation ---
@st.cache_data # Cache the data loading and preprocessing
def load_and_prepare_data():
    iris = load_iris()
    X = iris.data
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Apply PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42) # Set random state for reproducibility
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, iris.feature_names # Return PCA data and original feature names

X_pca, feature_names = load_and_prepare_data()

# --- Sidebar Configuration ---
st.sidebar.header("Configure Clustering")
k = st.sidebar.slider(
    "Select number of clusters (k)",
    min_value=2,
    max_value=10,
    value=4,  # Default value matching the image
    step=1
)

# --- Main Panel ---
# Set title - Using markdown for potential emoji/styling
st.markdown("## 🔍 K-Means Clustering App with Iris Dataset")
st.markdown(f"Visualizing clustering results for **k = {k}** using the Iris dataset (reduced to 2D via PCA).")


# --- K-Means Clustering ---
# Perform K-Means clustering with the selected k
kmeans = KMeans(
    n_clusters=k,
    init='k-means++', # Standard initialization
    n_init=10,        # Run multiple times with different centroids seeds
    random_state=42   # For reproducibility
    )
# Fit the model and predict clusters on the PCA data
# (Alternatively, fit on X_scaled and visualize on X_pca colored by those clusters)
y_kmeans = kmeans.fit_predict(X_pca)
cluster_centers_pca = kmeans.cluster_centers_ # Get cluster centers in PCA space

# --- Visualization ---
st.subheader(f"Clusters (2D PCA Projection)")

fig, ax = plt.subplots(figsize=(10, 6)) # Create figure and axes for better control

# Scatter plot for each cluster
for i in range(k):
    ax.scatter(
        X_pca[y_kmeans == i, 0],  # x-coordinates for cluster i
        X_pca[y_kmeans == i, 1],  # y-coordinates for cluster i
        label=f'Cluster {i}',     # Label for the legend
        alpha=0.8                 # Slightly transparent points
    )

# Plot cluster centers (optional but helpful)
# ax.scatter(
#     cluster_centers_pca[:, 0], cluster_centers_pca[:, 1],
#     s=200,                     # Size of the marker
#     c='red',                  # Color of the marker
#     marker='X',                # Marker style
#     label='Centroids'
# )


ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_title(f"K-Means Clustering with k={k} (Iris Dataset - PCA)")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

# Display the plot in Streamlit
st.pyplot(fig)

# --- Footer/Manage App ---
# Placeholder for the "Manage app" button often seen in Streamlit Cloud deployments
# This is usually part of the platform, not the app code itself.
# You could add a simple divider or footer text if desired:
st.divider()
st.caption("Streamlit K-Means Demo")

