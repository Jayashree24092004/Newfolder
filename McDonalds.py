import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Step 1: Load data
df = pd.read_csv(r"C:\Users\lenovo\OneDrive\Desktop\Market_segmentation\mcdonalds.csv")
print(df.columns)
print(df.shape)
print(df.head(3))

# Step 2: Convert first 11 columns to binary matrix (Yes -> 1, No -> 0)
MD_x = (df.iloc[:, 0:11] == "Yes").astype(int)

# Step 3: Column means
print(np.round(MD_x.mean(axis=0), 2))

# Step 4: PCA
pca = PCA()
MD_pca = pca.fit(MD_x)

# Step 5: PCA summary (explained variance)
print("Explained Variance Ratio:", np.round(pca.explained_variance_ratio_, 3))
print("PCA Components:\n", np.round(pca.components_, 1))

# Step 6: Plotting PCA projections
pca_scores = pca.transform(MD_x)
plt.figure(figsize=(8, 6))
plt.scatter(pca_scores[:, 0], pca_scores[:, 1], color='grey', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection of McDonalds Dataset')

# Step 7: Plotting variable vectors (like projAxes)
for i, (x, y) in enumerate(pca.components_[:2].T):
    plt.arrow(0, 0, x * 3, y * 3, color='red', head_width=0.05)
    plt.text(x * 3.2, y * 3.2, df.columns[i], color='blue', ha='center')


plt.grid(True)
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.show()

