Convolutional Neural Network (CNN) for CIFAR-10 Classification

This project implements a Convolutional Neural Network (CNN) for classifying images in the CIFAR-10 dataset using the PyTorch library.
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.
The task is to classify each image into one of the following classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

Introduction to Convolutional Neural Networks (CNNs):
Convolutional Neural Networks (CNNs) are a class of deep neural networks, most commonly applied to analyzing visual imagery.
They are widely used for image recognition, classification, segmentation, and many other tasks in computer vision.
CNNs are composed of multiple layers of convolutional and pooling operations, followed by fully connected layers for classification.
They have shown remarkable performance in various computer vision tasks due to their ability to automatically learn hierarchical features from raw pixel data.

About PyTorch:
PyTorch is an open-source machine learning library developed by Facebook's AI Research lab.
It provides tensor computation (similar to NumPy) with strong GPU acceleration, deep neural networks built on a tape-based autograd system, and dynamic computational graphs.
PyTorch is widely used for research and production in various domains, including computer vision, natural language processing, and reinforcement learning.
Overview of the Code

Model Architecture:
The CNN architecture used in this project consists of 5 convolutional layers followed by fully connected layers for classification.
Each convolutional layer is followed by a ReLU activation function and max-pooling operation to extract and downsample features from the input images.
The model architecture is as follows:

    Conv1: 32 filters, kernel size 3x3
    Conv2: 64 filters, kernel size 3x3
    Conv3: 128 filters, kernel size 3x3
    Conv4: 256 filters, kernel size 3x3
    Conv5: 512 filters, kernel size 3x3
    Fully connected layers: 120 units, 75 units, and 10 units for the output classes.

Dataset and DataLoader:
The CIFAR-10 dataset is used for training, validation, and testing. It is split into a training set (80%) and a validation set (20%) for model evaluation during training.
Additionally, a separate test set is used to evaluate the final performance of the trained model.

Training and Evaluation:
The model is trained using adam optimizer with a learning rate of 0.001.
During training, the loss function is the cross-entropy loss, and accuracy metrics are computed for both training and validation sets.
After training, the model is evaluated on the test set to assess its generalization performance.

Running the Code:
To run the code, make sure you have Python installed along with the necessary libraries listed in the requirements.txt file.
You can train the model by executing the provided Python script FirstCNN.py or by running the code in a Jupyter notebook environment.