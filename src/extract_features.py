# -*- coding: utf-8 -*-
"""
Script for loading an environmental sound dataset suchs as the Construction 
Site (CS), UrbanSound8K (US8K), ESC-10, and ESC-50 datasets and extracting 
spectrogram, phasogram, scalogram, wavelet phasogram, and MFCCgram after 
resampling audio to 22,050 Hz, as described in:
    
- Michele Scarpiniti, Saud Hussain, Wangyi Pu, Aurelio Uncini, Yong-Cheol~Lee, 
"EfficientAudioNet: Enhancing Environmental Sound Classification through Data 
Fusion of Multiple Audio Representations", in 2025 International Joint Conference 
on Neural Networks (IJCNN 2025), Rome, Italy, June 30 - July 05, 2025.


@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""


import os
import numpy as np
import librosa
import cv2
import features as ft


f_res = 22050  # resample rate


# CS dataset
data_folder = '../Datasets/CS/'
save_folder = '../Data/CS/'

# US8K dataset
# data_folder = '../Datasets/US8K/'
# save_folder = '../Data/US8K/'

# ESC-10 dataset
# data_folder = '../Datasets/ESC10/'
# save_folder = '../Data/ESC10/'

# ESC-50 dataset
# data_folder = '../Datasets/ESC50/'
# save_folder = '../Data/ESC50/'


sets = ['Training/', 'Test/']


# Main loop
directory = data_folder
audio_directories = os.listdir(directory)
audio_directories.sort()

c = 0  # Class label
feat_S  = []   # For spectrogram
feat_P  = []   # For phasogram
feat_W  = []   # For scalogram
feat_Wp = []   # For wavelet phasogram
feat_M  = []   # For MFCCgram
labels  = []   # For labels


for s in sets:
    directory = data_folder + s
    print("Directory: ", directory)
    audio_directories = os.listdir(directory)
    print("Directory: ", audio_directories)
    audio_directories.sort()

    c = 0  # Class label
    feat_S  = []   # For spectrogram
    feat_P  = []   # For phasogram
    feat_W  = []   # For scalogram
    feat_Wp = []   # For wavelet phasogram
    feat_M  = []   # For MFCCgram
    labels  = []   # For labels

    for d in audio_directories:
        if d == '.DS_Store' or not os.path.isdir(os.path.join(directory, d)):
            continue  
        path_directories = directory + d
        file_list = os.listdir(path_directories)
        N = len(file_list)
        n = 0

        for f in file_list:
            file_name = path_directories + '/' + f

            # Load wave file
            x, fs = librosa.load(file_name, sr=None)
            x = librosa.resample(x, orig_sr=fs, target_sr=f_res)

            # Extract features
            S, P = ft.extract_spectrogram(x, hop_length=1, n_fft=1024)
            W, Q = ft.extract_scalogram(x, wavelet='morlet')
            M    = ft.extract_MFCCgram(S, n_mfcc=250)

            # Resize features to 224x224
            S = cv2.resize(S, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
            P = cv2.resize(P, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
            W = cv2.resize(W, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
            Q = cv2.resize(Q, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
            M = cv2.resize(M, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)

            # Normalize features
            S = ft.scale(S)
            P = ft.scale(P)
            W = ft.scale(W)
            Q = ft.scale(Q)
            M = ft.scale(M)

            # Transform as an integer image
            S = np.array(255*S, dtype = 'uint8')
            P = np.array(255*P, dtype = 'uint8')
            W = np.array(255*W, dtype = 'uint8')
            Q = np.array(255*Q, dtype = 'uint8')
            M = np.array(255*M, dtype = 'uint8')

            # Set label
            y = c

            # Append features and labels
            feat_S.append(S)
            feat_P.append(P)
            feat_W.append(W)
            feat_Wp.append(Q)
            feat_M.append(M)
            labels.append(y)

            # Print advancement
            n += 1
            if (n % 100):
                print("\r{}: {}%".format(d, round(100*n/N, 1)), end='')


        c += 1
        # print("Processed folder: ", d)
        print("\r{}: {}%".format(d, 100.0), end='\n')


    # Saving data
    save_path = save_folder + s

    np.save(save_path + 'spectrograms.npy', feat_S)
    np.save(save_path + 'phasogram.npy', feat_P)
    np.save(save_path + 'scalograms.npy', feat_W)
    np.save(save_path + 'wavphasogram.npy', feat_Wp)
    np.save(save_path + 'MFCCgram.npy', feat_M)
    np.save(save_path + 'label.npy', labels)

    print("Done for folder: ", s)


print("Done!")
