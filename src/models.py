# -*- coding: utf-8 -*-
"""
Script for the definition of the backbone models based on the EfficientNet-B0 
used to implement the Environmental Sound Classification (ESC) algorithms 
exploiting Early Fusion (EF), Intermediate Fusion (IF), and Late Fusion (LF) 
strategies of spectrograms, phasograms, scalograms, wavelet phasograms, and 
MFCCgrams for the classification of audio recordings related to the Construction 
Site (CS), UrbanSound8K (US8K), and ESC datasets, as described in:
    
- Michele Scarpiniti, Saud Hussain, Wangyi Pu, Aurelio Uncini, Yong-Cheol~Lee, 
"EfficientAudioNet: Enhancing Environmental Sound Classification through Data 
Fusion of Multiple Audio Representations", in 2025 International Joint Conference 
on Neural Networks (IJCNN 2025), Rome, Italy, June 30 - July 05, 2025.


@author: Saud Hussain and Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

import tensorflow as tf
from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Concatenate
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

tf.random.set_seed(42)



# Define the model with EfficientNet-B0 and custom layers
def create_pretrained_model(input_shape, num_classes, LR=0.0001):
    base_model = EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = True
    # Freeze the base model
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)  # Use Global Average Pooling
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model



# Function to create a base model based on the EfficientNet-B0
def create_shared_base_model(input_shape):
    base_model = EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = True
    # Freeze the base model
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    return base_model


# Function to create a base model based on the EfficientNet-B0
# def create_shared_base_model(input_shape):
#     base_model = EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet')
#     base_model.trainable = False
    
#     return base_model




# Function to create a branch using the shared base model EfficientNet-B0
def create_branch(input_shape, base_model, branch_name):
    inputs = Input(shape=input_shape, name=f'{branch_name}_input')
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D(name=f'{branch_name}_gap')(x)
    
    return inputs, x



# Function fto create a model with intermediate fusion
def create_intermediate_fusion_model(input_shape, num_classes, LR=0.0001):
    shared_base_model = create_shared_base_model(input_shape)
    
    # Combined model with intermediate fusion
    branch_names = ['Branch1', 'Branch2', 'Branch3', 'Branch4', 'Branch5']
    branches = [create_branch(input_shape, shared_base_model, name) for name in branch_names]

    # Concatenate the outputs of the branches
    concatenated = Concatenate([branch[1] for branch in branches], name='concatenated')
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.001), name='dense')(concatenated)
    x = Dropout(0.5, name='dropout')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)

    # Model
    model = Model(inputs=[branch[0] for branch in branches], outputs=outputs, name='intermediate_fusion_model')
    
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model



# Function to define the used callbacks
def define_callbacks(dataset, model_name):
    es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
    mc = ModelCheckpoint(f'{dataset}_{model_name}_best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
    rop = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5, verbose=1)
    
    return [es, mc, rop]



