#!/usr/bin/env python3
"""
Script to find the user at sample_index = 10 from XGBoost test set
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize Spark
spark = SparkSession.builder.appName("FindUser").getOrCreate()

# Load the data (same as in notebook)
data = spark.read.parquet("data/cleaned_churn_data.parquet")

# Create user features (same as in notebook)
user_features = (
    data.groupBy("userId")
    .agg(
        F.first("churn_flag").alias("churn_flag"),
        F.countDistinct("sessionId").alias("num_sessions"),
        F.sum(F.when(F.col("page") == "NextSong", 1).otherwise(0)).alias("num_songs"),
        F.sum(F.when(F.col("page") == "Thumbs Up", 1).otherwise(0)).alias("thumbs_up"),
        F.sum(F.when(F.col("page") == "Thumbs Down", 1).otherwise(0)).alias("thumbs_down"),
        F.sum(F.when(F.col("page") == "Add to Playlist", 1).otherwise(0)).alias("add_playlist"),
        ( (F.max("ts") - F.min("ts")) / (1000 * 60 * 60 * 24) ).alias("active_days")
    )
)

# Convert to pandas
feature_cols = [
    "num_sessions",
    "num_songs", 
    "thumbs_up",
    "thumbs_down",
    "add_playlist",
    "active_days"
]

pdf = user_features.select(*feature_cols, "churn_flag", "userId").toPandas()
X = pdf[feature_cols].fillna(0)
y = pdf["churn_flag"].astype(int)
user_ids = pdf["userId"]

# Recreate the same train-test split as used in XGBoost
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Get the corresponding user IDs for the test set
user_ids_train, user_ids_test = train_test_split(
    user_ids, test_size=0.3, random_state=42, stratify=y
)

# Find the user at sample_index = 10
sample_index = 10
user_id_at_index_10 = user_ids_test.iloc[sample_index]

print(f"🔍 User at sample_index = {sample_index}:")
print(f"User ID: {user_id_at_index_10}")
print(f"Actual churn status: {y_test.iloc[sample_index]} ({'CHURNED' if y_test.iloc[sample_index] == 1 else 'DID NOT CHURN'})")

# Get the full user information
user_info = pdf[pdf["userId"] == user_id_at_index_10]
print(f"\n📊 Full user information:")
print(user_info[["userId"] + feature_cols + ["churn_flag"]].to_string(index=False))

# Show the feature values that were used in SHAP
print(f"\n🎯 Feature values used in SHAP analysis:")
feature_values = X_test.iloc[sample_index]
for i, feature in enumerate(feature_cols):
    print(f"  {feature}: {feature_values.iloc[i]}")

print(f"\n💡 This user's SHAP explanation shows how these {len(feature_cols)} features contributed to their churn prediction!")

spark.stop()

