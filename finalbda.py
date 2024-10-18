import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

# Create a Spark session
spark = SparkSession.builder.appName("VehicleAcceptabilityClassification").getOrCreate()

# Load the CSV file
df = spark.read.csv("C:/Users/Asarv/Downloads/car.csv", header=False, inferSchema=True)

# Assign column names
column_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "acceptability"]
df = df.toDF(*column_names)

# Convert categorical variables to numerical
categorical_cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
stages = []

for col in categorical_cols:
    stringIndexer = StringIndexer(inputCol=col, outputCol=col + "_index")
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[col + "_onehot"])
    stages += [stringIndexer, encoder]

# Assemble features
assembler = VectorAssembler(inputCols=[c + "_onehot" for c in categorical_cols], outputCol="features")
stages += [assembler]

# Create and fit the pipeline
pipeline = Pipeline(stages=stages)
model = pipeline.fit(df)
df_encoded = model.transform(df)

# Perform K-means clustering
kmeans = KMeans(k=5, seed=1)
kmeans_model = kmeans.fit(df_encoded.select("features"))

# Make predictions
predictions = kmeans_model.transform(df_encoded)

# Evaluate clustering
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print(f"Silhouette with squared euclidean distance: {silhouette}")

# Prepare data for deep learning
label_indexer = StringIndexer(inputCol="acceptability", outputCol="label")
df_indexed = label_indexer.fit(df_encoded).transform(df_encoded)

# Convert to Pandas for TensorFlow
pandas_df = df_indexed.select("features", "label").toPandas()

X = np.array(pandas_df["features"].tolist())
y = pandas_df["label"].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build deep learning model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Clustering visualization
pandas_predictions = predictions.select("features", "prediction", "acceptability").toPandas()
plt.figure(figsize=(10, 8))
sns.scatterplot(x=pandas_predictions["features"].apply(lambda x: x[0]), 
                y=pandas_predictions["features"].apply(lambda x: x[1]), 
                hue=pandas_predictions["prediction"])
plt.title("Vehicle Clusters")
plt.show()

# Stop the SparkSession
spark.stop()