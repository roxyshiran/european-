import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv("your_dataset.csv")  # Replace with your dataset

# Data Preprocessing
data['Age'] = pd.cut(data['Age'], bins=[18, 24, 34, 44, 54, 64, np.inf], labels=['18-24', '25-34', '35-44', '45-54', '55-64', '65+'])
data['Country'] = data['Country'].astype(str)

# Handle Missing Values
data = data.dropna(subset=['Physical_Activity', 'Sedentary_Behavior', 'Sleep_Hours'])

# Normalize numeric data for pattern analysis
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['Physical_Activity', 'Sedentary_Behavior', 'Sleep_Hours']])

# **Clustering: KMeans**
kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust the number of clusters as needed
data['Cluster'] = kmeans.fit_predict(data_scaled)

# Visualize Clusters
sns.scatterplot(x='Physical_Activity', y='Sedentary_Behavior', hue='Cluster', data=data, palette='tab10')
plt.title("Clustering of Physical Activity and Sedentary Behavior")
plt.show()

# **Dimensionality Reduction: PCA**
pca = PCA(n_components=2)
pca_results = pca.fit_transform(data_scaled)

# Add PCA components back to the dataset
data['PCA1'] = pca_results[:, 0]
data['PCA2'] = pca_results[:, 1]

# Plot PCA results
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data, palette='tab10')
plt.title("PCA of Physical Activity, Sedentary Behavior, and Sleep")
plt.show()

# **Country-Level Patterns**
# Aggregating data by country
country_summary = data.groupby('Country')[['Physical_Activity', 'Sedentary_Behavior', 'Sleep_Hours']].mean().reset_index()

# Heatmap of country-level patterns
sns.heatmap(country_summary.set_index('Country'), annot=True, cmap='coolwarm')
plt.title("Country-Level Averages of Activity, Behavior, and Sleep")
plt.show()

# **Age Group Patterns**
sns.boxplot(x='Age', y='Physical_Activity', hue='Cluster', data=data)
plt.title("Physical Activity Patterns Across Age Groups")
plt.show()

sns.boxplot(x='Age', y='Sedentary_Behavior', hue='Cluster', data=data)
plt.title("Sedentary Behavior Patterns Across Age Groups")
plt.show()

sns.boxplot(x='Age', y='Sleep_Hours', hue='Cluster', data=data)
plt.title("Sleep Patterns Across Age Groups")
plt.show()
