# CIFAR-10 Neural Network Classifier (NumPy Implementation)

This project implements a basic neural network from scratch using NumPy to classify images from the CIFAR-10 dataset. It focuses on 3 specific classes: **Airplane**, **Automobile**, and **Bird**.

## 📂 Dataset
- CIFAR-10 dataset (10 classes, 32x32 RGB images)
- Only classes 0 (Airplane), 1 (Automobile), and 2 (Bird) are used
- Place the CIFAR-10 batch files in a folder named `cifar10_data/`

## 🧠 Model Architecture
- **Input layer:** 3072 nodes (32x32x3 flattened image)
- **Hidden layer:** 128 neurons with ReLU activation
- **Output layer:** 3 neurons (softmax for classification)

## 🔁 Training Details
- Optimizer: Mini-batch Gradient Descent
- Loss: Cross-Entropy
- Learning Rate: Starts at 0.01 with decay per epoch
- Epochs: 100
- Batch Size: 64

## 📈 Outputs
- `training_curves.png`: Training loss and accuracy per epoch
- `confusion_matrix.png`: Test set confusion matrix
- `analysis.txt`: Summary of performance and improvement suggestions
- `cifar10_nn_report_short.txt`: Short project report


