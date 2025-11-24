# Facial Emotion Detection Project Report

## 1. Introduction
This project aims to build a facial emotion detection system using PyTorch and OpenCV. The goal is to classify facial expressions into one of seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

## 2. Methodology

### 2.1 Data
The project uses the Facial Expression Dataset from Kaggle.
- **Input**: 48x48 pixel grayscale images.
- **Preprocessing**:
    - Images are converted to tensors.
    - Normalized with mean `[0.5, 0.5, 0.5]` and standard deviation `[0.25, 0.25, 0.25]`.
    - Data augmentation techniques like `RandomHorizontalFlip` and `RandomRotation` are applied to the training set.

### 2.2 Models
Two different approaches were implemented and compared:

#### Model 1: Custom ResNet-18
- A ResNet-18 architecture built from scratch.
- **Structure**:
    - Initial Convolutional Layer (7x7 kernel, stride 2).
    - Max Pooling.
    - 4 Residual Layers (each containing 2 Basic Blocks).
    - Adaptive Average Pooling and a Fully Connected Layer.
- **Input Channels**: 3 (converted from grayscale to pseudo-RGB for compatibility or consistency).

#### Model 2: Transfer Learning (ResNet-18)
- A pre-trained ResNet-18 model from `torchvision.models`.
- **Modifications**:
    - The final Fully Connected (FC) layer was replaced to match the number of classes (7) in the dataset.
    - Pre-trained weights were used to leverage features learned from ImageNet.

## 3. Training
- **Loss Function**: CrossEntropyLoss.
- **Optimizer**: Adam.
- **Hyperparameters**:
    - Batch Size: 128.
    - Epochs: 32.
    - Learning Rate: 0.008 (with OneCycleLR scheduler).
    - Gradient Clipping: 0.1.
    - Weight Decay: 0.0001.

## 4. Results
Both models were trained for 32 epochs. The final accuracy on the test dataset is as follows:

| Model | Description | Final Test Accuracy |
| :--- | :--- | :--- |
| **Model 1** | Custom ResNet-18 | **55.22%** |
| **Model 2** | Transfer Learning (ResNet-18) | **57.06%** |

*Note: The transfer learning approach yielded slightly better results compared to the custom implementation.*

## 5. Conclusion
The project successfully demonstrates facial emotion detection using Deep Learning. While the custom ResNet-18 provided a solid baseline, leveraging transfer learning with a pre-trained ResNet-18 model resulted in improved accuracy. Further improvements could be achieved by fine-tuning hyperparameters, using deeper architectures, or employing more advanced data augmentation techniques.
