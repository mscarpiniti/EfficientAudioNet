# -*- coding: utf-8 -*-
"""
Script for the definition of functions for extracting spectrogram, phasogram, 
scalogram, wavelet phasogram, and MFCCgram from waveform data, as described in:
    
- Michele Scarpiniti, Saud Hussain, Wangyi Pu, Aurelio Uncini, Yong-Cheol~Lee, 
"EfficientAudioNet: Enhancing Environmental Sound Classification through Data 
Fusion of Multiple Audio Representations", in 2025 International Joint Conference 
on Neural Networks (IJCNN 2025), Rome, Italy, June 30 - July 05, 2025.


@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""


import os
import numpy as np
import librosa
from ssqueezepy import cwt, stft



# Load a wave data
def load_wave_data(audio_dir, file_name):
    file_path = os.path.join(audio_dir, file_name)
    x, fs = librosa.load(file_path, sr=None)

    return x, fs



# Extract the STFT spectrogram
def extract_spectrogram(x, hop_length=1, n_fft=2048):
    X = librosa.stft(x, hop_length=hop_length, n_fft=n_fft)
    S = np.abs(X)    # spectrogram
    P = np.angle(X)  # Phasogram
    # print("Shape of spectrogram:", S.shape)

    return S, P



# Extract the STFT spectrogram using the ssqueezepy library
def extract_spectrogram_2(x, fs=22050, n_fft=2048):
    X = stft(x, fs=fs, n_fft=n_fft)[::-1]
    S = np.abs(X)    # spectrogram
    P = np.angle(X)  # Phasogram
    # print("Shape of spectrogram:", S.shape)

    return S, P



# Extract the MFCC-gram
def extract_MFCCgram(S, sr=22050, n_mfcc=128):
    # M = librosa.feature.mfcc(y=x, sr=sr)
    SM = librosa.feature.melspectrogram(S=S**2, sr=sr, n_mels=n_mfcc)
    M = librosa.feature.mfcc(S=librosa.power_to_db(SM, ref=np.max), sr=sr, n_mfcc=n_mfcc)
    # print("Shape of MFCC-gram:", M.shape)

    return M



# Extract the CWT scalogram
def extract_scalogram(x, fs=22050, wavelet='morlet'):
    X, scales = cwt(x, wavelet=wavelet, fs=fs)
    W = np.abs(X)    # Scalogram
    P = np.angle(X)  # Phasogram
    # print("Shape of scalogram:", W.shape)

    return W, P



# Scale matrix in range [0, 1]
def scale(matrix):
    # Perform min-max scaling
    min_val = np.min(matrix)
    max_val = np.max(matrix)

    scaled_matrix = (matrix - min_val) / (max_val - min_val)

    return scaled_matrix
