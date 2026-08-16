Overview
Tom & Jerry Image Classification Challenge
Goal
The goal of this competition is to develop a computer vision and machine learning model capable of detecting specific cartoon characters in image frames.

Participants are required to build a multi-class image classification model that can accurately identify the presence of 'Tom' and 'Jerry' in frames extracted from the classic animated series.

This competition provides an opportunity to apply image preprocessing, data augmentation, deep learning model development, and evaluation techniques to solve a challenging and noisy computer vision problem.

Start

10 hours ago
Close

14 hours to go
Description
Problem Description
Character recognition in video frames is a fundamental challenge in computer vision and multimedia analytics. With the vast amount of video content available, developing intelligent systems to automate tagging, content moderation, and analysis has become essential.

In this competition, participants will develop machine learning models to classify image frames based on the presence of characters.

The provided dataset contains frames extracted from the classic Tom & Jerry cartoon series. Participants must process these images, extract meaningful visual features, and develop predictive models capable of handling severely imbalanced and noisy data.

Dataset
The dataset contains image frames extracted at 1 frame per second (1 FPS) from episodes of the Tom & Jerry show, designed for computer vision research and deep learning experimentation.

The image filenames have been completely anonymized to prevent data leakage.

The target variable is:

appearance

where the classes are:

0 - Neither character is present
1 - Only Tom is present in the frame
2 - Only Jerry is present in the frame
3 - Both Tom and Jerry are present
Participants must use the provided training dataset to learn visual patterns associated with the characters and predict the exact appearance category for unseen test images.

Competition Task
Participants must:

Explore and analyze the provided image dataset.
Perform necessary image preprocessing (e.g., resizing, normalization, augmentation).
Develop a machine learning or deep learning classification model.
Generate predictions for the unseen test dataset.
Submit predictions in the required format.
Allowed Approaches
Participants may use any suitable machine learning techniques, including:

Convolutional Neural Networks (CNNs)
Transfer Learning with Pre-trained Models (e.g., ResNet, EfficientNet, VGG)
Vision Transformers (ViTs)
Ensemble Learning Methods
Participants are highly encouraged to experiment with advanced data augmentation and model optimization techniques.

Important Notes
Class Imbalance: The training dataset contains a severe imbalanced class distribution.
The test dataset is 100% clean and accurately labeled, ensuring a fair evaluation.
Solutions should prioritize generalization performance rather than overfitting the noisy and imbalanced training data.
Evaluation
Submissions are evaluated using the Macro F1-score between the predicted appearance labels and the actual appearance values.

The Macro F1-score is selected because the training dataset contains a heavy class imbalance. Accuracy alone may not represent model performance effectively, as a model could simply predict the majority class and achieve a moderately high score. The Macro F1-score calculates the metric for each class independently and finds their unweighted mean, ensuring that minority classes are equally important.

The F1-score is calculated as:


where:

Precision measures the proportion of correctly identified cases among all predicted cases for a class.
Recall measures the proportion of actual cases for a class that were successfully detected.
Submission File
For every image in the test dataset, participants must predict the exact appearance category.

The submission file should contain two columns: filename and appearance.

Citation
Lakmana Thabrew and Marshad Musni. OctWave 3.0 - Kaggle Challenge 02. https://kaggle.com/competitions/oct-wave-3-0-kaggle-challenge-02, 2026. Kaggle.
