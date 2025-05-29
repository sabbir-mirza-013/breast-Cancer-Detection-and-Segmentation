# Multitask U-Net for Breast Cancer Segmentation and Classification

This repository contains an implementation of a **Multitask U-Net** model using TensorFlow/Keras for simultaneous segmentation and classification of breast ultrasound images.

---

## Dataset

- Breast ultrasound images stored in one directory.
- Corresponding JSON annotation files containing segmentation masks and metadata stored in another directory.
- Images are preprocessed by resizing, normalization, and optional noise reduction.

---

## Features

- **Shared encoder** architecture extracting features for both tasks.
- **U-Net style decoder** with skip connections for pixel-wise segmentation.
- **Classification head** with global average pooling and dense layers.
- Custom **Dice loss** and **Dice coefficient** for segmentation.
- Joint training with combined weighted losses for segmentation and classification.
- Data generator implemented with Keras `Sequence` for efficient batch loading and preprocessing.

---

## Requirements

- Python 3.x  
- TensorFlow 2.x  
- OpenCV  
- NumPy  
- Matplotlib  
- scikit-learn  
- Pillow (PIL)

---

## Usage

1. **Prepare Dataset**  
   Organize your dataset with separate folders for images and JSON annotations.

2. **Train Model**  
   Train using the provided training and validation split generators.

3. **Save Model**  
   Save the trained model using:
   ```python
   model.save('multitask_unet_model.h5')
