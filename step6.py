import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans


df = pd.read_csv(r"C:\Users\lenovo\OneDrive\Desktop\Market_segmentation\Newfolder\mcdonalds.csv")
X = (df.iloc[:, 0:11] == "Yes").astype(int)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
kmeans = KMeans(n_clusters=4, random_state=1234)
labels = kmeans.fit_predict(X)

import seaborn as sns
from scipy.spatial.distance import pdist

dist_cols = pdist(X.T, metric='euclidean')
linkage_cols = linkage(dist_cols, method='ward')
dendro = dendrogram(linkage_cols, labels=X.columns, orientation='right')
plt.title("Hierarchical Clustering of Attributes")
plt.xlabel("Distance")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x=labels, palette="Set2")
plt.title("Cluster Membership Count (KMeans k=4)")
plt.xlabel("Cluster")
plt.ylabel("Number of respondents")
plt.xticks([0, 1, 2, 3])
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Set2', s=50)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("KMeans Clusters Projected on PCA Space")
plt.grid(True)
for i, col in enumerate(X.columns):
    vec = pca.components_.T[i]  
    x, y = vec * 3  
    plt.arrow(0, 0, x, y, color='red', alpha=0.5)
    plt.text(x * 1.15, y * 1.15, col, color='red', ha='center', va='center')

plt.tight_layout();
plt.show();
