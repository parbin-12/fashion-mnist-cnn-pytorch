# Fashion-MNIST Image Classification using Convolutional Neural Networks (PyTorch)

## Project Overview

This project implements a Convolutional Neural Network (CNN) from scratch using PyTorch to classify images from the Fashion-MNIST dataset.

The objective is to learn image features automatically through convolutional layers and classify clothing items into one of ten categories.

The model was trained using Stochastic Gradient Descent (SGD), Batch Normalization, Dropout Regularization, and Cross-Entropy Loss.

---

# Dataset

Fashion-MNIST is a benchmark computer vision dataset consisting of grayscale images of fashion products.

### Dataset Characteristics

* Number of Classes: 10
* Image Size: 28 × 28 pixels
* Training Samples: 60,000
* Testing Samples: 10,000
* Color Channels: 1 (Grayscale)

### Classes

| Label | Class       |
| ----- | ----------- |
| 0     | T-shirt/Top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle Boot  |

---

# Data Preprocessing

The following preprocessing steps were performed:

1. Load dataset using Pandas.
2. Separate features and labels.
3. Split dataset into training and testing sets (80-20).
4. Normalize pixel values from [0,255] to [0,1].
5. Convert NumPy arrays into PyTorch tensors.
6. Reshape images into:

```text
(Batch Size, 1, 28, 28)
```

to make them compatible with CNN layers.

---

# CNN Architecture

## Input Layer

```text
1 × 28 × 28
```

Grayscale image.

---

## Feature Extraction Block

### Convolution Layer 1

```text
Conv2D(1 → 32)
Kernel Size = 3×3
Padding = Same
```

Output:

```text
32 × 28 × 28
```

Followed by:

* ReLU Activation
* Batch Normalization
* Max Pooling (2×2)

After pooling:

```text
32 × 14 × 14
```

---

### Convolution Layer 2

```text
Conv2D(32 → 64)
Kernel Size = 3×3
Padding = Same
```

Output:

```text
64 × 14 × 14
```

Followed by:

* ReLU Activation
* Batch Normalization
* Max Pooling (2×2)

After pooling:

```text
64 × 7 × 7
```

---

## Classification Block

### Flatten Layer

Converts:

```text
64 × 7 × 7
```

into:

```text
3136 Features
```

---

### Fully Connected Layer 1

```text
3136 → 128
```

* ReLU
* Dropout(0.4)

---

### Fully Connected Layer 2

```text
128 → 64
```

* ReLU
* Dropout(0.4)

---

### Output Layer

```text
64 → 10
```

Produces logits for the ten Fashion-MNIST classes.

---

# Hyperparameters

| Parameter     | Value            |
| ------------- | ---------------- |
| Epochs        | 15               |
| Batch Size    | 32               |
| Learning Rate | 0.01             |
| Optimizer     | SGD              |
| Weight Decay  | 1e-4             |
| Loss Function | CrossEntropyLoss |
| Dropout       | 0.4              |

---

# Training Results

## Loss Progression

| Epoch | Loss   |
| ----- | ------ |
| 1     | 0.6504 |
| 2     | 0.3790 |
| 3     | 0.3220 |
| 4     | 0.2840 |
| 5     | 0.2610 |
| 6     | 0.2418 |
| 7     | 0.2235 |
| 8     | 0.2097 |
| 9     | 0.1943 |
| 10    | 0.1845 |
| 11    | 0.1726 |
| 12    | 0.1609 |
| 13    | 0.1559 |
| 14    | 0.1453 |
| 15    | 0.1407 |

The loss decreases consistently across all epochs, indicating stable convergence and successful learning.

---

# Model Performance

## Training Accuracy

```text
96.96%
```

## Testing Accuracy

```text
92.03%
```

The model demonstrates strong generalization performance on unseen data while maintaining high training accuracy.

---

# Key Concepts Demonstrated

* Custom PyTorch Dataset Creation
* DataLoader Usage
* Convolutional Neural Networks
* Batch Normalization
* Dropout Regularization
* SGD Optimization
* Forward Propagation
* Backpropagation
* Image Classification
* Model Evaluation

---

# Installation

Install required packages:

```bash
pip install -r requirements.txt
```

---

# Running the Project

```bash
python CNN.py
```

---

# Future Improvements

* Hyperparameter Tuning with Optuna
* Learning Rate Scheduling
* AdamW Optimizer
* Data Augmentation
* Transfer Learning with MobileNetV2 or ResNet18
* MLflow Experiment Tracking

---

# Author

Praveen Kumar Yadav

B.Tech Computer Science and Engineering

Dr. B.R. Ambedkar National Institute of Technology, Jalandhar
