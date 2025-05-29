#Multitask U-Net for Breast Cancer Segmentation and Classification

This repository contains a multitask U-Net model implemented in TensorFlow/Keras designed for simultaneous breast ultrasound image segmentation and image-level classification (normal, benign, malignant). The model uses a shared encoder with separate segmentation and classification heads.


Dataset
Breast ultrasound images with corresponding JSON annotations containing segmentation masks and labels.

Images are preprocessed by resizing, normalization, and optional noise reduction.

Features
Shared encoder architecture with U-Net style segmentation decoder.

Classification head with global average pooling and dense layers.

Custom Dice loss and Dice coefficient metric for segmentation.

Supports multitask learning with combined loss and metrics.

Data generator implemented using Keras Sequence for efficient loading and preprocessing.     

Usage
Train the model using the provided training and validation splits.

Save and load the trained model easily for inference or further training.

Predict and visualize segmentation masks and classification results on unseen images.

Requirements
Python 3.x

TensorFlow 2.x

OpenCV

NumPy

Matplotlib

scikit-learn

PIL

How to Run
Prepare your dataset directory with images and JSON annotations.

Train the model by running train.py or the main notebook.

Save the trained model using model.save().

Load the model and run inference on new images.

Use evaluation scripts to generate performance metrics and visualizations.

