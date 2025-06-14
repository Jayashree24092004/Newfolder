import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.utils import resample

# Load data
df = pd.read_csv("C:/Users/lenovo/OneDrive/Desktop/Market_segmentation/mcdonalds.csv")

X = (df.iloc[:, :11] == "Yes").astype(int)

# Try k from 2 to 8
inertias = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=1234)
    km.fit(X)
    inertias.append(km.inertia_)

plt.plot(range(2, 9), inertias, marker='o')
plt.xlabel("Number of segments")
plt.ylabel("Inertia")
plt.title("KMeans: Inertia vs Segments")
plt.show()
