import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
os.environ["OMP_NUM_THREADS"] = "1"

# Load the dataset
df = pd.read_csv("C:/Users/lenovo/OneDrive/Desktop/Market_segmentation/Newfolder/mcdonalds.csv")

# Clean 'Like' column
like_map = {
    "I hate it!-5": 1,
    "I dislike it-2": 2,
    "It is okay3": 3,
    "I like it!+4": 4,
    "I love it!+5": 5
}
df["Like.n"] = df["Like"].map(like_map)

# Clean VisitFrequency
visit_mapping = {
    "Every three months": 3,
    "Once a month": 2,
    "Once a week": 1,
    "Multiple times a week": 0.5,
    "Never": 5
}
df["VisitFreqNum"] = df["VisitFrequency"].map(visit_mapping)

# Select numeric features and drop rows with missing values
features = df[["Like.n", "VisitFreqNum"]].dropna()
valid_index = features.index

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df.loc[valid_index, "segment"] = kmeans.fit_predict(scaled_features)

# Calculate grouped statistics
visit = df.groupby("segment")["VisitFreqNum"].mean()
like = df.groupby("segment")["Like.n"].mean()
female_ratio = df.groupby("segment")["Gender"].apply(lambda x: (x == "Female").mean())

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(visit, like, s=1000 * female_ratio, alpha=0.6, color='skyblue', edgecolor='black')

for i, (x, y) in enumerate(zip(visit, like)):
    plt.text(x, y, str(i), fontsize=12, ha='center', va='center', color='black')

plt.xlim(2, 4.5)
plt.ylim(-3, 3)
plt.xlabel("Mean Visit Frequency (lower = more frequent)")
plt.ylabel("Mean Like Score")
plt.title("Cluster Summary: Visit vs Like (Bubble = % Female)")
plt.grid(True)
plt.show()
