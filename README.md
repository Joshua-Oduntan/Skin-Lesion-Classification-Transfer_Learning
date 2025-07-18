# Skin Lesion Classification with Transfer Learning
This project applies transfer learning techniques using pre-trained convolutional neural networks (CNNs) such as MobileNet, VGG19, and InceptionV3 to classify skin lesions from image data. It is designed to assist in the early detection of skin-related diseases like melanoma, basal cell carcinoma, and benign keratosis using deep learning.

## Dataset
The dataset used is from Kaggle:
https://www.kaggle.com/datasets/vinayjayanti/skin-lesion-image-classification

Multi-class dataset containing high-quality images of skin lesions.

Images are labeled by lesion type.

## Features
Implements three popular CNN architectures: MobileNet, VGG19, and InceptionV3.

Uses data augmentation and preprocessing to enhance model generalization.

Tracks performance across multiple simulation runs per model.

Includes model saving, confusion matrix, and classification reports for analysis.

## Tech Stack
Python

TensorFlow / Keras

Scikit-learn

Matplotlib / Seaborn

Mlxtend

Google Colab (for cloud training)

## Model Workflow
Data Preparation

Unzips and organizes image dataset into training and testing folders.

Uses ImageDataGenerator for preprocessing and real-time augmentation.

Model Training

Initializes MobileNet, VGG19, and InceptionV3 using pretrained ImageNet weights.

Adds custom classifier head and performs fine-tuning.

Runs multiple simulations per model for performance consistency.

Evaluation

Evaluates models using accuracy, confusion matrix, and classification report.

Visualizes performance metrics with plots.

Model Saving

Saves trained models for future use or deployment.

## Results Summary
Each model was trained and tested on the same dataset for comparative analysis. Performance metrics included:

Accuracy

Confusion Matrix

Precision, Recall, and F1-score (via classification report)


## How to Run
Clone the repository

Upload the dataset to Google Drive

Open Transfer_Learning_for_skin_lesions.ipynb in Google Colab

Run all cells sequentially

Evaluate results and download saved model(s)

## Learnings & Contributions
Gained experience in using pre-trained deep learning models for medical image classification.

Explored fine-tuning techniques and their impact on model performance.

Understood the importance of data preprocessing and augmentation in healthcare datasets
