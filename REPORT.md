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

## 5. Realtime Emotion Detection

After training the models, realtime emotion detection was implemented to detect and classify facial emotions from a live webcam feed. The implementation uses the following approach:

### 5.1 Face Detection
- **Haar Cascade Classifier**: OpenCV's `haarcascade_frontalface_default.xml` is used to detect faces in the video frames.
- **Detection Parameters**: 
  - `scaleFactor=1.1`: Image pyramid scale factor for multi-scale detection.
  - `minNeighbors=5`: Minimum number of neighboring rectangles to keep a detection.

### 5.2 Frame Preprocessing Pipeline
For each detected face, the following preprocessing steps are applied:
1. **Face Extraction**: The detected face region is cropped from the grayscale frame with additional padding (`y-70:y+h+10, x:x+w`).
2. **Color Conversion**: Grayscale image is converted to 3-channel BGR format using `cv2.cvtColor()` to match the model's input requirements.
3. **Resizing**: The cropped face is resized to 48×48 pixels to match the training data dimensions.
4. **Tensor Transformation**: The image is converted to a PyTorch tensor and normalized.
5. **Batch Reshaping**: The tensor is reshaped from `(C, H, W)` to `(1, C, H, W)` for batch processing.
6. **GPU Transfer**: The tensor is moved to the GPU device for faster inference.

### 5.3 Emotion Prediction
The trained model performs inference on the preprocessed frame:
- The model outputs class probabilities for all 7 emotion categories.
- `torch.max()` is used to select the emotion with the highest confidence.
- The predicted emotion label is retrieved from the `class_names` list.

### 5.4 Temporal Smoothing
To reduce prediction jitter and create a smoother user experience:
- **Moving Window Averaging**: A moving window of size 24 frames stores recent predictions.
- **Voting Mechanism**: The most frequent emotion in the window is selected using `np.bincount()` and `np.argmax()`.
- **Window Update**: After each prediction, the window is rolled left using `np.roll()` and the new prediction is added.

### 5.5 Visualization
The live feed displays:
- A bounding box around the detected face.
- The current predicted emotion as text overlay using `cv2.putText()`.
- A grayscale video feed for better face detection performance.

### 5.6 Exit Condition
The realtime detection loop runs continuously until the user presses the 'q' key, at which point the webcam connection is released and all windows are closed.

## 6. Conclusion
The project successfully demonstrates facial emotion detection using Deep Learning. While the custom ResNet-18 provided a solid baseline, leveraging transfer learning with a pre-trained ResNet-18 model resulted in improved accuracy. The realtime detection implementation extends the trained model's capabilities to process live video streams, making it practical for real-world applications. Further improvements could be achieved by fine-tuning hyperparameters, using deeper architectures, or employing more advanced data augmentation techniques.
