import librosa
import numpy as np
import pandas as pd

chunk_size = 3  # seconds

def extract_features(audio_data, sample_rate, feature_columns):
    samples_per_chunk = sample_rate * chunk_size
    chunks = [
        audio_data[i:i + samples_per_chunk]
        for i in range(0, len(audio_data), int(samples_per_chunk))
    ]

    return pd.concat(
        extract_features_for_chunk(chunk, sample_rate, feature_columns)
        for chunk in chunks
    )


def extract_features_for_chunk(audio_data, sample_rate, feature_columns):
    features = {
        'length': len(audio_data),
        'chroma_stft_mean': np.mean(librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)),
        'chroma_stft_var': np.var(librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)),
        'rms_mean': np.mean(librosa.feature.rms(y=audio_data)),
        'rms_var': np.var(librosa.feature.rms(y=audio_data)),
        'spectral_centroid_mean': np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)),
        'spectral_centroid_var': np.var(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)),
        'spectral_bandwidth_mean': np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)),
        'spectral_bandwidth_var': np.var(librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)),
        'rolloff_mean': np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)),
        'rolloff_var': np.var(librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)),
        'zero_crossing_rate_mean': np.mean(librosa.feature.zero_crossing_rate(y=audio_data)),
        'zero_crossing_rate_var': np.var(librosa.feature.zero_crossing_rate(y=audio_data)),
        'harmony_mean': np.mean(librosa.effects.harmonic(y=audio_data)),
        'harmony_var': np.var(librosa.effects.harmonic(y=audio_data)),
        'perceptr_mean': np.mean(librosa.effects.percussive(y=audio_data)),
        'perceptr_var': np.var(librosa.effects.percussive(y=audio_data)),
        'tempo': librosa.beat.tempo(y=audio_data, sr=sample_rate)[0]
    }

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate)
    for i, mfcc in enumerate(mfccs, start=1):
        features[f'mfcc{i}_mean'] = np.mean(mfcc)
        features[f'mfcc{i}_var'] = np.var(mfcc)

    for key in features:
        features[key] = float(features[key])  # Convert each float32 to native float

    return prepare_features_for_model(features, feature_columns)

def prepare_features_for_model(extracted_features, feature_columns):
    df_features = pd.DataFrame([extracted_features])
    df_features = df_features[feature_columns]
    return df_features
