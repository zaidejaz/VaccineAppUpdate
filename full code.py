import tensorflow as tf
import numpy as np
import pandas as pd
from libreco.data import random_split, DatasetPure
from libreco.algorithms import NCF
from libreco.evaluation import evaluate

# Load and preprocess the data
data = pd.read_csv("ratings.csv")
data.columns = ["user", "item", "label", "time"]
train_data, eval_data, test_data = random_split(data, multi_ratios=[0.8, 0.1, 0.1])

# Build Collaborative Filtering Dataset
train_data_cf, data_info_cf = DatasetPure.build_trainset(train_data)
eval_data_cf = DatasetPure.build_evalset(eval_data)
test_data_cf = DatasetPure.build_testset(test_data)

# Train NCF for Collaborative Filtering
ncf = NCF(
    task="rating",
    data_info=data_info_cf,
    loss_type="cross_entropy",
    embed_size=16,
    n_epochs=10,
    lr=1e-3,
    batch_size=2048,
    num_neg=1,
)
ncf.fit(
    train_data_cf,
    neg_sampling=False,
    verbose=2,
    eval_data=eval_data_cf,
    metrics=["loss"],
)

# Load and preprocess the content data
content_data = pd.read_csv("content.csv")  # Replace with your content data
content_data.columns = ["filename", "chroma_stft", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "rmse", "zero_crossing_rate", "name", "surah", "juz"]  # Adjust column names accordingly

# Calculate the number of unique filenames in the content-based dataset
content_item_size = len(content_data["filename"].unique())

# Build Content-Based Dataset
train_data_cb, data_info_cb = DatasetPure.build_trainset(train_data)
eval_data_cb = DatasetPure.build_evalset(eval_data)
test_data_cb = DatasetPure.build_testset(test_data)

# Define and Train LSTM for Content-Based Recommendation
lstm_model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(input_dim=content_item_size, output_dim=16),
        tf.keras.layers.LSTM(units=32),
        tf.keras.layers.Dense(units=16, activation="relu"),
        tf.keras.layers.Dense(units=8, activation="relu"),
    ]
)
lstm_model.compile(optimizer="adam", loss="mse")

# Convert content features to numpy array
content_features = content_data.loc[content_data["filename"].isin(train_data_cb.X["item"]), ["chroma_stft", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "rmse", "zero_crossing_rate", "filename", "name", "surah", "juz"]].values
train_data_cb.X["content"] = np.array([np.array(features) for features in content_features])

# Train LSTM model
lstm_model.fit(
    x=train_data_cb.X["content"],
    y=train_data_cb.Y,
    epochs=10,
    batch_size=2048,
    validation_data=(eval_data_cb.X["content"], eval_data_cb.Y),
)

# Weighted Hybrid Recommendation
weights = {
    "collaborative": 0.7,  # Adjust the weights according to your preference
    "content-based": 0.3,
}

# Predictions from Collaborative Filtering (NCF)
collab_predictions = ncf.predict(user=5755, item=110)

# Convert content features for the specific item to numpy array
item_content_features = content_data.loc[content_data["filename"] == 110, ["chroma_stft", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "rmse", "zero_crossing_rate", "filename", "name", "surah", "juz"]].values
item_content_input = np.array([np.array(item_content_features)])

# Predictions from Content-Based (LSTM)
content_predictions = lstm_model.predict(item_content_input)

# Weighted Hybrid Combination
hybrid_predictions = (weights["collaborative"] * collab_predictions) + (weights["content-based"] * content_predictions)

# Print hybrid_predictions
print(hybrid_predictions)