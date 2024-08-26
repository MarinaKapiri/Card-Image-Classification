# Card-Image-Classification
This code trains a PyTorch model to classify playing cards. It sets up a custom dataset, builds a model using EfficientNet, and trains it with cross-entropy loss. The code includes data loading, training, and testing, along with visualization of predictions.

The code provides a complete pipeline for training, validating, and testing a deep learning model to classify images of playing cards using PyTorch and the TIMM library. The pipeline involves several key steps:

Dataset Preparation: The PlayingCardDataset class is defined, which loads images using the ImageFolder class and applies transformations like resizing and tensor conversion. Data is then loaded into PyTorch using DataLoader for efficient batching and shuffling.

Model Definition: A custom neural network SimpleCardClassifier is defined, using an EfficientNet backbone from the TIMM library. The model is modified to output the desired number of classes (53 for this case).

Training and Validation: The model is trained using cross-entropy loss and the Adam optimizer. A learning rate scheduler and early stopping are implemented to prevent overfitting. Training and validation losses are tracked and plotted over epochs.

Inference and Visualization: The model is tested on unseen images, and predictions are visualized with probability distributions for each class.

Utility Functions: Various functions for preprocessing images, predicting outcomes, and visualizing results are provided to evaluate the model's performance on new data.

This setup is designed for training a classifier that can distinguish between different playing cards in a dataset, handling the entire process from data loading to model evaluation.
