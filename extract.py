import librosa
import numpy as np
import pandas as pd

def extract_features(audio_data, sample_rate):
    # Calculate various features of the audio file
    features = {
        'length': librosa.get_duration(y=audio_data, sr=sample_rate),
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

    # Calculate the mean and variance of the MFCCs
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate)
    for i, mfcc in enumerate(mfccs, start=1):
        features[f'mfcc{i}_mean'] = np.mean(mfcc)
        features[f'mfcc{i}_var'] = np.var(mfcc)

    for key in features:
        features[key] = float(features[key])  # Convert each float32 to native float

    return features

import pandas as pd

df_structure = pd.read_csv('Data/features_3_sec.csv', nrows=0)

def prepare_features_for_model(extracted_features):
    df_features = pd.DataFrame([extracted_features])
    df_features = df_features[df_structure.columns.drop(['filename', 'label'])]
    return df_features

# Prepare the data for the model
# X_for_prediction = prepare_features_for_model(extracted_features)

# Now, `X_for_prediction` can be fed into the model for prediction


# audio_data, sample_rate = librosa.load('/home/lumenalux/Downloads/cell-phone-ring-fx_112bpm.wav', sr=None)
# print(extract_features(audio_data, sample_rate))
