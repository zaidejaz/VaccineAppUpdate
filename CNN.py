import pandas as pd
import numpy as np
import tensorflow as tf

# Load and preprocess the content data
content_data = pd.read_csv("content.csv")  # Replace with the actual filename
content_data.columns = ["filename", "chroma_stft", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "rmse", "zero_crossing_rate", "name", "surah", "juz"]  # Adjust column names accordingly

# Convert content features to numpy array
content_features = content_data[["chroma_stft", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "rmse", "zero_crossing_rate"]].values
content_features = np.array([np.array(features) for features in content_features])

# Create the LSTM model
lstm_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(content_data), output_dim=16, input_length=6),
    tf.keras.layers.LSTM(units=32),
    tf.keras.layers.Dense(units=16, activation="relu"),
    tf.keras.layers.Dense(units=8, activation="relu"),
    tf.keras.layers.Dense(units=1, activation="linear")
])

lstm_model.compile(optimizer="adam", loss="mse")

# Train the LSTM model
lstm_model.fit(
    x=content_features,
    y=content_data["filename"],  # Replace with your target variable
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# Make predictions using the trained model
test_features = content_features[:10]  # Example: Using the first 10 samples as test data
predictions = lstm_model.predict(test_features)

# Print the predictions
print(predictions)