# -*- coding: utf-8 -*-
"""
Script for implementing the Intermediate Fusion (IF) strategy of spectrograms, 
phasograms, scalograms, wavelet phasograms, and MFCCgrams for the classification 
of audio recordings related to the Construction Site (CS), UrbanSound8K (US8K), 
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

features, labels = ut.load_all_feature_data(training_folder, feature_files, label_file, concat=False)
print('Number of instances:', len(labels))



# Split the data into train and validation sets
X_train, X_val, y_train, y_val = ut.split_train_val(features, labels)
print('Shape of training set', X_train[0].shape)
print('Shape of validation set', X_val[0].shape)


input_shape = X_train[0].shape[1:]
num_classes = y_train.shape[1]
model = models.create_intermediate_fusion_model(input_shape, num_classes, LR)

# Free memory after loading data
del features, labels



# Train the model
history = model.fit(
    [X_train[i] for i in range(len(X_train))], y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,    
    validation_data=([X_val[i] for i in range(len(X_val))], y_val),
    callbacks=models.define_callbacks(dataset, 'IF')
)


# Plot the training and validation accuracy and loss
ut.plot_accuracy(history, 'Intermediate Fusion')
ut.plot_loss(history, 'Intermediate Fusion')


# Save the fine-tuned model
model.save(dataset + '_IF.keras')


# Free memory after fine-tuning
del X_train
del y_train



# Test the model
class_names = ut.get_labels(dataset)

# Load test features and labels
testing_folder = '../Data/' + dataset + '/Test/'
test_features, test_labels = ut.load_all_feature_data(testing_folder, feature_files, label_file, concat=False)
print('Shape of test set', test_features.shape)


# Load the fine-tuned model
# model = load_model("CS_IF.keras")


# Evaluate on test data
# test_loss, test_accuracy = model.evaluate([test_features_subset[i] for i in range(len(test_features_subset))], test_labels_subset)
# print(f"Test loss: {test_loss:.4f}%")
# print(f"Test accuracy on of the test data: {test_accuracy * 100:.2f}%")



# Generate the classification report
y_pred_probs = model.predict([test_features[i] for i in range(len(test_features))])
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(test_labels, axis=1)


# Print classification report
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)


# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
# ut.plot_confusion_matrix(conf_matrix, class_names, 'Intermediate Fusion')
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
disp.plot(cmap='Blues')

