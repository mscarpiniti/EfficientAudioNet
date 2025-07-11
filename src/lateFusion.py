# -*- coding: utf-8 -*-
"""
Script for implementing the Late Fusion (LF) strategy of spectrograms, phasograms,
scalograms, wavelet phasograms, and MFCCgrams for the classification of audio
recordings related to the Construction Site (CS), UrbanSound8K (US8K),
Environmental Sound CLassification (ESC10 and ESC50) datasets, as described in:

- Michele Scarpiniti, Saud Hussain, Wangyi Pu, Aurelio Uncini, Yong-Cheol Lee, 
"EfficientAudioNet: Enhancing Environmental Sound Classification through Data
Fusion of Multiple Audio Representations", in 2025 International Joint Conference
on Neural Networks (IJCNN 2025), Rome, Italy, June 30 - July 05, 2025.


@author: Saud Hussain and Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

import numpy as np
import utils as ut
import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
# from tensorflow.keras.models import load_model



# Select the dataset to be used
dataset = 'CS'
# dataset = 'US8K'
# dataset = 'ESC10'
# dataset = 'ESC50'


# Set the hyper-parameters
EPOCHS = 30
BATCH_SIZE = 16
LR = 0.0001


# Load features
training_folder = '../Data/' + dataset + '/Training/'
feature_files = [
    'spectrograms.npy',
    'phasogram.npy',
    'scalograms.npy',
    'wavphasogram.npy',
    'MFCCgram.npy'
]
label_file = 'label.npy'


# List to store trained models for each feature set
trained_models = []


# Loop through each feature set and train the model using the full dataset
for feature_file in feature_files:
    feature_data, feature_labels = ut.load_single_feature_data(training_folder, feature_file, label_file, concat=False)

    # Train/validation split for this feature set
    X_train, X_val, y_train, y_val = train_test_split(feature_data, feature_labels, test_size=0.2, stratify=feature_labels, random_state=42)

    # Free memory after loading data
    del feature_data, feature_labels


    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]

    # Create and train the model
    model, _ = models.create_pretrained_model(input_shape, num_classes, LR)
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=models.define_callbacks(dataset, 'LF_'+feature_file.split('.')[0])
    )

    # Append the trained model to the list
    trained_models.append(model)

    # Free memory after each model's fine-tuning
    del X_train, y_train, X_val, y_val



# Plot training and validation accuracy and loss for the last model
ut.plot_accuracy(history, 'LF (Last Model)')
ut.plot_loss(history, 'LF (Last Model)')



# Test the model
class_names = ut.get_labels(dataset)


# Load test features and labels
testing_folder = '../Data/' + dataset + '/Test/'
test_features, test_labels = ut.load_all_feature_data(testing_folder, feature_files, label_file, concat=False)
print('Shape of test set', test_features.shape)


# Make predictions for late fusion
y_pred_probs = ut.combine_predictions(test_features, trained_models)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(test_labels, axis=1)


# Classification report
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
# ut.plot_confusion_matrix(conf_matrix, class_names, 'Late Fusion')

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
disp.plot(cmap='Blues')
