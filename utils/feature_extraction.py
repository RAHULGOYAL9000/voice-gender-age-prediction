#!/usr/bin/env python
# coding: utf-8

# In[2]:

# utils/feature_extraction.py

import numpy as np
import librosa

def extract_features(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)

    # Ensure audio is not silent
    if np.all(y == 0):
        raise ValueError("Silent audio detected.")

    # Extract features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    # Aggregate statistics
    features = np.hstack([
        np.mean(mfccs, axis=1),
        np.std(mfccs, axis=1),
        np.mean(chroma, axis=1),
        np.std(chroma, axis=1),
        np.mean(zcr),
        np.std(zcr),
        np.mean(rolloff),
        np.std(rolloff),
        np.mean(centroid),
        np.std(centroid),
        np.mean(bandwidth),
        np.std(bandwidth),
    ])

    return features


# In[ ]:




