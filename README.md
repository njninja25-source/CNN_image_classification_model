# CNN_image_classification_model
This project implements a Convolutional Neural Network (CNN) in PyTorch to classify images from the CIFAR‑10 dataset (60,000 32×32 color images across 10 classes). The model is trained using supervised learning with cross‑entropy loss

# Model
The model is a Convolutional Neural Network (CNN) built with PyTorch, designed to classify images from the CIFAR-10 dataset. It extracts hierarchical features through convolutional and pooling layers, then maps them to class probabilities using fully connected layers.

# Convolutional Layers
1st Convolutional Block

Input: 32 × 32 × 3 (RGB image)

Conv2d(3 → 32, kernel_size=3, padding=1)

Activation: ReLU

Pooling: MaxPool2d(2,2) → reduces dimension to 16 × 16 × 32

2nd Convolutional Block

Input: 16 × 16 × 32

Conv2d(32 → 64, kernel_size=3, padding=1)

Activation: ReLU

Pooling: MaxPool2d(2,2) → reduces dimension to 8 × 8 × 64

3rd Convolutional Block

Input: 8 × 8 × 64

Conv2d(64 → 128, kernel_size=3, padding=1)

Activation: ReLU

Pooling: MaxPool2d(2,2) → reduces dimension to 4 × 4 × 128

# Fully Connected Layers
Flattening

Output from last conv block: 4 × 4 × 128 = 2048 features

Hidden Layer 1

Linear(2048 → 256)

Activation: ReLU

Output Layer

Linear(256 → 10)

Produces logits for 10 CIFAR-10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

# Training Setup
Loss Function: CrossEntropyLoss (includes softmax internally)

Optimizer: Adam

Batch Size: 64

Epochs: 10

# Key Notes for Reference
Each convolutional layer uses padding=1 to preserve spatial dimensions before pooling.

MaxPool2d(2,2) halves the spatial resolution at each stage.

