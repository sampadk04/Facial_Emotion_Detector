# Facial Emotion Detector

A facial emotion detection system built from scratch using PyTorch and OpenCV.

![Project Preview](project-asset.png)

## Emotion Detector Overview

### Data
- Facial expression [dataset](http://www.kaggle.com/dataset/de270025c781ba47a3a6d774a0d670452bfb4dc9d2d6b13740cdb0c17aa7bf2b) from Kaggle.

### Classifiers
- **Custom ResNet-18 Implementation**
- **Transfer Learning**: Pre-trained ResNet-18 on a subset of the ImageNet dataset.

### Realtime Rendering (from WebCam)
- **Preprocessing**: OpenCV
- **Frame Facial Detection**: Haar Cascade Frontal Face Detector

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd Facial_Emotion_Detector
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook Facial_Emotion_Detection.ipynb
    ```

2.  Run the cells to train the models or use the pre-trained weights (if available) for detection.

## Project Structure

```
Facial_Emotion_Detector/
├── data/                       # Dataset directory
├── models/                     # Saved models
├── sample_images/              # Images for README
├── Facial_Emotion_Detection.ipynb # Main project notebook
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Results on the Test Data

![Results](sample_images/results.jpeg)

## Sample Realtime Detection from WebCam

### Happy Face
<img src="sample_images/happy_face.jpeg" alt="Happy Face" height="300" width="450"/>

### Neutral Face
<img src="sample_images/neutral_face.jpeg" alt="Neutral Face" height="300" width="450"/>
