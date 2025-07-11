# -*- coding: utf-8 -*-
"""
Script for the definition of useful functions used for the data fusion strategies
based on the EfficientNet-B0 implementing the Environmental Sound
Classification (ESC) algorithms to exploit Early Fusion (EF), Intermediate 
Fusion (IF), and Late Fusion (LF) strategies of spectrograms, phasograms,
scalograms, wavelet phasograms, and MFCCgrams for the classification of audio
recordings related to the Construction Site (CS), UrbanSound8K (US8K), and ESC
datasets, as described in:

- Michele Scarpiniti, Saud Hussain, Wangyi Pu, Aurelio Uncini, Yong-Cheol~Lee,
"EfficientAudioNet: Enhancing Environmental Sound Classification through Data
Fusion of Multiple Audio Representations", in 2025 International Joint Conference
on Neural Networks (IJCNN 2025), Rome, Italy, June 30 - July 05, 2025.


@author: Saud Hussain and Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split



# Function for loading a single feature and labels
def load_single_feature_data(training_folder, feature_file, label_file):
    labels = np.load(os.path.join(training_folder, label_file))
    # Convert feature labels to categorical
    labels = to_categorical(labels)

    feature_data = np.load(os.path.join(training_folder, feature_file))
    # Add channel dimension and convert to 3-channel input for EfficientNetB0
    feature_data = np.repeat(feature_data[..., np.newaxis], 3, axis=-1)

    return feature_data, labels



# Function for loading features and labels
def load_all_feature_data(training_folder, feature_files, label_file, concat=True):
    labels = np.load(os.path.join(training_folder, label_file))
    # Convert feature labels to categorical
    labels = to_categorical(labels)

    feature_data = [np.load(os.path.join(training_folder, file)) for file in feature_files]
    if concat:
        feature_data = np.concatenate(feature_data, axis=-1)
    # Add channel dimension and convert to 3-channel input for EfficientNetB0
    feature_data = np.repeat(feature_data[..., np.newaxis], 3, axis=-1)

    return feature_data, labels



# Function for combining the predictionin late fusion
def combine_predictions(feature_data, trained_models):
    test_predictions = []

    for model, data in zip(trained_models, feature_data):
        # Get predictions from the corresponding trained model
        predictions = model.predict(data)
        test_predictions.append(predictions)

    # Late fusion by averaging predictions across all feature-based models
    test_predictions = np.mean(test_predictions, axis=0)

    return test_predictions



# Function to load and split features and labels
def load_split_data(training_folder, feature_files, label_file):
    # Load labels
    labels = np.load(os.path.join(training_folder, label_file))
    # Convert feature labels to categorical
    labels = to_categorical(labels)

    features = [np.load(os.path.join(training_folder, file)) for file in feature_files]

    # Initialize lists to hold train, validation, and test data
    X_train, X_val, X_test = [], [], []
    y_train, y_val, y_test = [], [], []

    # Split the data into train-test-val for each feature set and labels
    for feature in features:
        # First split: train-test split
        X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(
            feature, labels, test_size=0.2, random_state=42)
        # Second split: train-validation split from the train set
        X_temp_train, X_temp_val, y_temp_train, y_temp_val = train_test_split(
            X_temp_train, y_temp_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2 for validation

        # Append to respective lists
        X_train.append(X_temp_train)
        X_val.append(X_temp_val)
        X_test.append(X_temp_test)
        y_train, y_val, y_test = y_temp_train, y_temp_val, y_temp_test

        # Add channel dimension for Conv2D layers and convert to 3 channels for each feature set
        X_train = [np.repeat(X[..., np.newaxis], 3, axis=-1) for X in X_train]
        X_val   = [np.repeat(X[..., np.newaxis], 3, axis=-1) for X in X_val]
        X_test  = [np.repeat(X[..., np.newaxis], 3, axis=-1) for X in X_test]

    return X_train, X_val, X_test, y_train, y_val, y_test



# Function to split the data into train and validation sets
def split_train_val(features, labels):
    X_train_split = [feature for feature in features]
    X_train, X_val, y_train, y_val = [], [], [], []

    for feature in X_train_split:
        X_train_feature, X_val_feature, y_train_split, y_val_split = train_test_split(
            feature, labels, test_size=0.2, random_state=42)
        X_train.append(X_train_feature)
        X_val.append(X_val_feature)
        y_train, y_val = y_train_split, y_val_split

    return X_train, X_val, y_train, y_val



# Function to plot the training and validation accuracy of the model
def plot_accuracy(history, fusion):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy - {fusion}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()



# Function to plot the training and validation loss of the model
def plot_loss(history, fusion):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Training and Validation Loss - {fusion}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()



# Function to plot the confusion matrix with Seaborn
def plot_confusion_matrix(cm, class_names, fusion):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {fusion}')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()



# Funtion to retrieve the class names of each dataset
def get_labels(dataset):
    if dataset == 'CS':
        labels = ['Concrete Mixer',
                  'Excavator CAT320',
                  'Compactor Ingersoll',
                  'BackhoeJD50DComp',
                  'Excavator Hitachi50']
    elif dataset == 'US8K':
        labels = ['Air conditioner',
                  'Car horn',
                  'Children playing',
                  'Dog bark',
                  'Drilling',
                  'Engine idling',
                  'Gun shot',
                  'Jackhammer',
                  'Siren',
                  'Street music']
    elif dataset == 'ESC10':
        labels = ['Chainsaw',
                  'Clock tick',
                  'Crackling fire',
                  'Crying baby',
                  'Dog',
                  'Helicopter',
                  'Rain',
                  'Rooster',
                  'Sea waves',
                  'Sneezing']
    elif dataset == 'ESC50':
        labels = ['Airplane','Breathing','Brushing teeth','Can opening',
                  'Car horn','Cat','Chainsaw','Chirping birds','Church bells',
                  'Clapping','Clock alarm','Clock tick','Coughing','Cow',
                  'Crackling fire','Crickets','Crow','Crying baby','Dog',
                  'Door - wood creaks','Door knock','Drinking - sipping',
                  'Engine','Fireworks','Footsteps','Frog','Glass breaking',
                  'Hand saw','Helicopter','Hen','Insects (flying)',
                  'Keyboard typing','Laughing','Mouse click','Pig',
                  'Pouring water','Rain','Rooster','Sea waves','Sheep','Siren',
                  'Sneezing','Snoring','Thunderstorm','Toilet flush','Train',
                  'Vacuum cleaner','Washing machine','Water drops','Wind']
    else:
        print("Dataset name is wrong! Select only one between: CS, US8K, ESC10 or ESC50")

    return labels
