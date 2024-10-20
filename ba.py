import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import PCA
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import when
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Create a Spark session
spark = SparkSession.builder.appName("VehicleAcceptabilityClassification").getOrCreate()

# Load the CSV file
df = spark.read.csv("C:/Users/Asarv/Downloads/car.csv", header=True, inferSchema=True)

# Combine 'good' and 'vgood' into one category
df = df.withColumn("Car_Acceptability", 
                   when(df["Car_Acceptability"].isin(["good", "vgood"]), "good_vgood")
                   .otherwise(df["Car_Acceptability"]))

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

# Create label indexer
label_indexer = StringIndexer(inputCol="Car_Acceptability", outputCol="label")
stages += [label_indexer]

# Create and fit the pipeline
pipeline = Pipeline(stages=stages)
model = pipeline.fit(df)
df_transformed = model.transform(df)

# Split data into training and testing sets
train_df, test_df = df_transformed.randomSplit([0.8, 0.2], seed=42)

# Perform clustering (using the existing code)
kmeans = KMeans(k=5, featuresCol="pca_features", seed=42)
kmeans_model = kmeans.fit(train_df)

# Make predictions for clustering
train_predictions = kmeans_model.transform(train_df)
test_predictions = kmeans_model.transform(test_df)

# Evaluate clustering
evaluator = ClusteringEvaluator(featuresCol="pca_features")
train_silhouette = evaluator.evaluate(train_predictions)
test_silhouette = evaluator.evaluate(test_predictions)
print(f"Train Silhouette: {train_silhouette}")
print(f"Test Silhouette: {test_silhouette}")

# Add Logistic Regression classifier
lr = LogisticRegression(featuresCol="pca_features", labelCol="label", maxIter=10)
lr_model = lr.fit(train_df)

# Make predictions for classification
train_class_predictions = lr_model.transform(train_df)
test_class_predictions = lr_model.transform(test_df)

# Evaluate the classification model
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(test_class_predictions)
print(f"Accuracy: {accuracy}")

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(test_class_predictions)
print(f"F1 Score: {f1_score}")

# Plotting the classification results
plt.figure(figsize=(12, 8))
pandas_predictions = test_class_predictions.select("pca_features", "prediction", "Car_Acceptability").toPandas()
pca_features = np.array(pandas_predictions["pca_features"].tolist())
scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], c=pandas_predictions["prediction"], cmap="viridis")
plt.colorbar(scatter)
plt.title('Vehicle Classification (Test Data)')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.show()

# Classification analysis
class_stats = pandas_predictions.groupby('prediction').agg({
    'Car_Acceptability': lambda x: x.value_counts().index[0],
    'pca_features': lambda x: np.mean(np.vstack(x), axis=0)
}).reset_index()

# Dynamically create column names based on the number of PCA components
pca_columns = [f'PC{i+1}_Mean' for i in range(pca_features.shape[1])]

# Print information about the shapes
print(f"Shape of class_stats: {class_stats.shape}")
print(f"Number of PCA columns: {len(pca_columns)}")
print(f"Current columns of class_stats: {class_stats.columns.tolist()}")

# Ensure we have the correct number of columns
if len(class_stats.columns) - 2 != len(pca_columns):
    print("Warning: Number of PCA columns doesn't match the data. Adjusting...")
    pca_columns = [f'PC{i+1}_Mean' for i in range(len(class_stats.columns) - 2)]

# Now set the column names
new_columns = ['Class', 'Dominant_Acceptability'] + pca_columns
class_stats.columns = new_columns

print(f"Final columns of class_stats: {class_stats.columns.tolist()}")
print(class_stats)

# Stop the SparkSession
spark.stop()
