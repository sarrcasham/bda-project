import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Create a Spark session
spark = SparkSession.builder.appName("VehicleAcceptabilityClassification").getOrCreate()

# Load the CSV file
df = spark.read.csv("C:/Users/Asarv/Downloads/car.csv", header=True, inferSchema=True)

# Convert categorical variables to numerical
categorical_cols = ["Buying_Price", "Maintenance_Price", "No_of_Doors", "Person_Capacity", "Size_of_Luggage", "Safety"]
stages = []

for col in categorical_cols:
    stringIndexer = StringIndexer(inputCol=col, outputCol=col + "_index")
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[col + "_onehot"])
    stages += [stringIndexer, encoder]

# Assemble features
assembler = VectorAssembler(inputCols=[c + "_onehot" for c in categorical_cols], outputCol="assembled_features")
stages += [assembler]

# Standardize features
scaler = StandardScaler(inputCol="assembled_features", outputCol="scaled_features")
stages += [scaler]

# Apply PCA
pca = PCA(k=2, inputCol="scaled_features", outputCol="pca_features")
stages += [pca]

# Create and fit the pipeline
pipeline = Pipeline(stages=stages)
model = pipeline.fit(df)
df_transformed = model.transform(df)

# Split data into training and testing sets
train_df, test_df = df_transformed.randomSplit([0.8, 0.2], seed=42)

# Try different numbers of clusters
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(k=k, featuresCol="pca_features", seed=42)
    model = kmeans.fit(train_df)
    predictions = model.transform(train_df)
    evaluator = ClusteringEvaluator(featuresCol="pca_features")
    silhouette = evaluator.evaluate(predictions)
    silhouette_scores.append(silhouette)

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.show()

# Choose the best number of clusters
best_k = silhouette_scores.index(max(silhouette_scores)) + 2

# Perform clustering with the best k
kmeans = KMeans(k=best_k, featuresCol="pca_features", seed=42)
model = kmeans.fit(train_df)

# Make predictions
train_predictions = model.transform(train_df)
test_predictions = model.transform(test_df)

# Evaluate clustering
evaluator = ClusteringEvaluator(featuresCol="pca_features")
train_silhouette = evaluator.evaluate(train_predictions)
test_silhouette = evaluator.evaluate(test_predictions)
print(f"Train Silhouette: {train_silhouette}")
print(f"Test Silhouette: {test_silhouette}")

# Convert to Pandas for visualization
pandas_predictions = test_predictions.select("pca_features", "prediction", "Car_Acceptability").toPandas()

# Extract PCA features
pca_features = np.array(pandas_predictions["pca_features"].tolist())

# Plotting the clustering results
plt.figure(figsize=(12, 8))
scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], c=pandas_predictions["prediction"], cmap="viridis")
plt.colorbar(scatter)
plt.title('Vehicle Clusters (Test Data)')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.show()

# Cluster analysis
cluster_stats = pandas_predictions.groupby('prediction').agg({
    'Car_Acceptability': lambda x: x.value_counts().index[0],
    'pca_features': lambda x: np.mean(np.vstack(x), axis=0)
}).reset_index()

# Dynamically create column names based on the number of PCA components
pca_columns = [f'PC{i+1}_Mean' for i in range(pca_features.shape[1])]
cluster_stats.columns = ['Cluster', 'Dominant_Acceptability'] + pca_columns

print(cluster_stats)

# Stop the SparkSession
spark.stop()
