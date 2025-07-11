# -*- coding: utf-8 -*-
"""
Script for partion an environmental sound dataset such as the Construction 
Site (CS), UrbanSound8K (US8K), ESC-10, and ESC-50 datasets and extracting
spectrogram, phasogram, scalogram, wavelet phasogram, and MFCCgram into frames
of suitable length, such as 5 sec, as described in:

- Michele Scarpiniti, Saud Hussain, Wangyi Pu, Aurelio Uncini, Yong-Cheol~Lee,
"EfficientAudioNet: Enhancing Environmental Sound Classification through Data
Fusion of Multiple Audio Representations", in 2025 International Joint Conference
on Neural Networks (IJCNN 2025), Rome, Italy, June 30 - July 05, 2025.


@author: Saud Hussain and Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

import os
import random
import librosa
import numpy as np
import soundfile as sf


# Function to split audio data into frames
def partitionate_audio(directory, seg_length_sec, training_path, testing_path):
    """
    Partition audio files into fixed-length segments and save them as training and testing datasets.

    Args:
        - directory (str): Path to the root dataset directory containing class subdirectories.
        - seg_length_sec (float): Segment length in seconds.
        - training_path (str): Path to save training data.
        - testing_path (str): Path to save testing data.
    """
    class_directories = sorted(os.listdir(directory))

    sample_rate = 44100  # Audio sample rate
    frame_size  = int(sample_rate * seg_length_sec)  # Frame size in samples
    hop = frame_size  # Non-overlapping segments

    for class_dir in class_directories:
        class_path = os.path.join(directory, class_dir)

        if not os.path.isdir(class_path):
            continue

        for audio_file in os.listdir(class_path):
            if audio_file == '.DS_Store':
                continue
            path_wav_file = os.path.join(class_path, audio_file)
            print(f"Loading: {path_wav_file}")

            try:
                audio, sr = librosa.load(path_wav_file, sr=sample_rate)
            except Exception as e:
                print(f"Error loading {path_wav_file}: {e}")
                continue

            # Pad audio if shorter than frame size
            if len(audio) < frame_size:
                audio = np.pad(audio, (0, frame_size - len(audio)), mode='constant')

            # Partition the audio into frames
            try:
                partitioned_audio = librosa.util.frame(audio, frame_length=frame_size, hop_length=hop)
            except Exception as e:
                print(f"Error partitioning {path_wav_file}: {e}")
                continue

            frames_number = partitioned_audio.shape[1]
            print(f"Partitioning {class_dir} into {frames_number} frames of {seg_length_sec} seconds")

            # Shuffle and split frames
            frame_indices = list(range(frames_number))
            random.shuffle(frame_indices)

            train_segm_len = int(0.75 * frames_number)
            l_train = frame_indices[:train_segm_len]
            l_test = frame_indices[train_segm_len:]

            # Save training frames
            for i in l_train:
                frame = partitioned_audio[:, i]
                path_new_folder = os.path.join(training_path, class_dir)
                path_new_audio = os.path.join(path_new_folder, f"{os.path.splitext(audio_file)[0]}_{i}.wav")

                os.makedirs(path_new_folder, exist_ok=True)
                sf.write(path_new_audio, frame, sr, 'PCM_16')

            # Save testing frames
            for i in l_test:
                frame = partitioned_audio[:, i]
                path_new_folder = os.path.join(testing_path, class_dir)
                path_new_audio = os.path.join(path_new_folder, f"{os.path.splitext(audio_file)[0]}_{i}.wav")

                os.makedirs(path_new_folder, exist_ok=True)
                sf.write(path_new_audio, frame, sr, 'PCM_16')




# Define paths for CS
CS_dataset_path  = "..Datasets/RAW/CS/"        # Path of the CS dataset
CS_training_path = "../Datasets/CS/Training/"  # Path to save CS training data
CS_testing_path  = "../Datasets/CS/Test/"      # Path to save CS test data



# Define paths for UrbanSound8K
# US8K_dataset_path  = "../Datasets/RAW/US8K/"      # Path of the US8K dataset
# US8K_training_path = "../Datasets/US8K/Training/" # Path to save US8K training data
# US8K_testing_path  = "../Datasets/US8K/Test/"     # Path to save US8K testing data



# Define paths for ESC-10
# ESC10_dataset_path  = "../Datasets/RAW/ESC10/"      # Path of the ESC-10 dataset
# ESC10_training_path = "../Datasets/ESC10/Training/" # Path to save ESC-10 training data
# ESC10_testing_path  = "../Datasets/ESC10/Test/"     # Path to save ESC-10 testing data



# Define paths for ESC-50
# ESC50_dataset_path  = "../Datasets/RAW/ESC50/"      # Path of the ESC-50 dataset
# ESC50_training_path = "../Datasets/ESC50/Training/" # Path to save ESC-50 training data
# ESC50_testing_path  = "../Datasets/ESC50/Test/"     # Path to save ESC-50 testing data



# Segment length in seconds
seg_length_sec = 5  # Standard segment length for all datasets



# Process CS dataset
print("Processing CS dataset...")
partitionate_audio(CS_dataset_path, seg_length_sec, CS_training_path, CS_testing_path)


# Process UrbanSound8K dataset
# print("\nProcessing UrbanSound8K dataset...")
# partitionate_audio(US8K_dataset_path, seg_length_sec, US8K_training_path, US8K_testing_path)


# Process ESC-10 dataset
# print("Processing ESC-10 dataset...")
# partitionate_audio(ESC10_dataset_path, seg_length_sec, ESC10_training_path, ESC10_testing_path)


# Process ESC-50 dataset
# print("Processing ESC-50 dataset...")
# partitionate_audio(ESC50_dataset_path, seg_length_sec, ESC50_training_path, ESC50_testing_path)
