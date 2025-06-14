import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load dataset
df = pd.read_csv(r"C:\Users\lenovo\OneDrive\Desktop\Market_segmentation\Newfolder\mcdonalds.csv")


# Dummy k4 cluster labels (replace with actual clustering result)
# For demonstration, using k-means to simulate clustering
from sklearn.cluster import KMeans

binary_cols = df.select_dtypes(include=['object']).columns[:11]
df[binary_cols] = df[binary_cols].apply(lambda col: col.astype('category').cat.codes)

kmeans = KMeans(n_clusters=4, random_state=1234)
df['k4'] = kmeans.fit_predict(df[binary_cols])

# Mosaic plot: Segment (k4) vs Like
plt.figure(figsize=(8, 6))
mosaic(df, ['k4', 'Like'])
plt.title('Segment vs Like')
plt.xlabel('Segment number')
plt.show()

# Mosaic plot: Segment (k4) vs Gender
plt.figure(figsize=(8, 6))
mosaic(df, ['k4', 'Gender'])
plt.title('Segment vs Gender')
plt.show()

# Create Like.n variable
df['Like.n'] = 6 - pd.to_numeric(df['Like'], errors='coerce')

# Binary classification target: is cluster 3?
df['target'] = (df['k4'] == 3).astype(int)

# Encode features
features = ['Like.n', 'Age', 'VisitFrequency', 'Gender']
df_encoded = pd.get_dummies(df[features], drop_first=True)

# Train decision tree
tree = DecisionTreeClassifier(max_depth=3, random_state=1234)
tree.fit(df_encoded, df['target'])

# Plot the decision tree
plt.figure(figsize=(14, 7))
plot_tree(tree, feature_names=df_encoded.columns, class_names=['Not Segment 3', 'Segment 3'], filled=True)
plt.show()
