# -*- coding: utf-8 -*-
"""
Script for implementing the Early Fusion (EF) strategy of spectrograms, phasograms, 
scalograms, wavelet phasograms, and MFCCgrams for the classification of audio 
recordings related to the Construction Site (CS), UrbanSound8K (US8K), 
Environmental Sound CLassification (ESC10 and ESC50) datasets, as described in:
    
- Michele Scarpiniti, Saud Hussain, Wangyi Pu, Aurelio Uncini, Yong-Cheol~Lee, 
"EfficientAudioNet: Enhancing Environmental Sound Classification through Data 
Fusion of Multiple Audio Representations", in 2025 International Joint Conference 
on Neural Networks (IJCNN 2025), Rome, Italy, June 30 - July 05, 2025.


@author: Saud Hussain and Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

import numpy as np
import utils as ut
import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
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


# Load dataset
training_folder = '../Data/' + dataset + '/Training/'
feature_files = [
    'spectrograms.npy',
    'phasogram.npy',
    'scalograms.npy',
    'wavphasogram.npy',
    'MFCCgram.npy'
]
label_file = 'label.npy'


features, labels = ut.load_all_feature_data(training_folder, feature_files, label_file)
print('Number of instances:', len(labels))


# Split data into training and validation sets only
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)
print('Shape of training set', X_train.shape)
print('Shape of validation set', X_val.shape)


# Free memory after loading data
del features, labels




# Define the model with EfficientNetB0 and custom layers
input_shape = X_train.shape[1:]
num_classes = y_train.shape[1]
model, _ = models.create_pretrained_model(input_shape, num_classes, LR)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=models.define_callbacks(dataset, 'EF')
)


# Plot the training and validation accuracy and loss
ut.plot_accuracy(history, 'Early Fusion')
ut.plot_loss(history, 'Early Fusion')


# Save the fine-tuned model
model.save(dataset + '_EF.keras')


# Free memory after fine-tuning
del X_train, X_val, y_train, y_val




# Test the model
class_names = ut.get_labels(dataset)


# Load testing data
testing_folder = '..Data/' + dataset + '/Test/'
test_features, test_labels = ut.load_all_feature_data(testing_folder, feature_files, label_file)
print('Shape of test set', test_features.shape)



# Load the fine-tuned model
# model = load_model('CS_EF.keras')


# Evaluate on test data
# test_loss, test_accuracy = model.evaluate(test_features, test_labels)
# print(f"Test loss: {test_loss:.4f}%")
# print(f"Test accuracy: {test_accuracy * 100:.2f}%")



# Generate predictions
y_pred_probs = model.predict(test_features)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(test_labels, axis=1)


# Print classification report
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)


# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
# ut.plot_confusion_matrix(conf_matrix, class_names, 'Early Fusion')
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
disp.plot(cmap='Blues')

