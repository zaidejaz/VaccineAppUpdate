import os
import pandas as pd
import librosa
import numpy as np

audio_directory = r"C:\Users\Haifa\AppData\Roaming\sa.edu.ksa.ayat\Local Store\audio\Nasser_Alqatami_29-30"
audio_features = []

for filename in os.listdir(audio_directory):
    file_path = os.path.join(audio_directory, filename)

    # Load the MP3 file using librosa
    audio, sr = librosa.load(file_path)

    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    rmse = librosa.feature.rms(y=audio)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)

    # Add the audio features to the list
    audio_features.append({
        'filename': filename,
        'chroma_stft': chroma_stft.mean(),
        'spectral_centroid': spectral_centroid.mean(),
        'spectral_bandwidth': spectral_bandwidth.mean(),
        'spectral_rolloff': spectral_rolloff.mean(),
        'rmse': rmse.mean(),
        'zero_crossing_rate': zero_crossing_rate.mean()
    })

# Convert the list into a pandas DataFrame
df = pd.DataFrame(audio_features)

# Save the DataFrame as a CSV file
df.to_csv("fd8.csv", index=False)