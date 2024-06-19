# Facial Emotion Detector

A facial emotion detection system built from scratch using PyTorch and OpenCV.

## Emotion Detector Overview

### Data
- Facial expression [dataset](http://www.kaggle.com/dataset/de270025c781ba47a3a6d774a0d670452bfb4dc9d2d6b13740cdb0c17aa7bf2b) from Kaggle.

### Classifiers
- **Custom ResNet-18 Implementation**
- **Transfer Learning**: Pre-trained ResNet-18 on a subset of the ImageNet dataset.

### Realtime Rendering (from WebCam)
- **Preprocessing**: OpenCV
- **Frame Facial Detection**: Haar Cascade Frontal Face Detector

## Results on the Test Data

![Results](sample_images/results.jpeg)

## Sample Realtime Detection from WebCam

### Happy Face
<img src="sample_images/happy_face.jpeg" alt="Happy Face" height="300" width="450"/>

### Neutral Face
<img src="sample_images/neutral_face.jpeg" alt="Neutral Face" height="300" width="450"/>
