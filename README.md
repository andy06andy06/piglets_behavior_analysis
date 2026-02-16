# Piglets Behavior Analysis System

This repository provides an integrated deep learning framework designed to automate the monitoring and behavioral analysis of piglets in nursery pens. The system covers the entire pipeline from raw video preprocessing and spatial calibration to object detection and complex action recognition.

## 🚀 Key Features

* **Action Recognition (CRNN)**: Implements a **Convolutional Recurrent Neural Network** using **EfficientNet** as a spatial feature encoder and **LSTM** as a temporal decoder. This architecture captures time-series information from video sequences to classify complex behaviors.
* **Object Detection & Tracking (YOLO & SORT)**: Leverages **YOLOv7/YOLO11** for individual piglet detection and the **SORT** algorithm for multi-object tracking to quantify movement metrics and spatial distribution.
* **Spatial Calibration (Perspective Warp)**: Includes robust tools for image rectification using perspective transforms tailored to specific pen layouts, ensuring consistent Regions of Interest (ROI) across different camera angles.
* **Automated Event Filtering**: Automatically identifies and extracts feeding or drinking segments by calculating the IoU (Intersection over Union) between detection bounding boxes and predefined functional zones.
* **Quantitative Analytics**: Built-in utilities to calculate the **Nearest Neighbor Index (NNI)** and movement histograms to analyze social interaction patterns and activity levels.

## 🛠️ Technical Details

### 1. Behavior Classification Categories

The system is trained to recognize 3 core behaviors:
`Feeding`, `Drinking`, `Aggression`.

### 2. Model Components

* **Encoder**: EfficientNet-B0 (pretrained), utilized for robust spatial feature extraction.
* **Decoder**: LSTM (512 hidden units), processing temporal sequences of 22 frames per segment (using a skip-frame strategy).
* **Detection Backbone**: Ultralytics YOLO11 (optimized for real-time nursery environment monitoring).

### 3. Data Preprocessing Pipeline

* `video2img17.py`: Handles perspective warping and frame extraction for spatial normalization.
* `XMLtoTXT.py`: Converts annotation files from XML format to YOLO-compliant TXT files (supporting oriented/rotated bounding boxes).
* `drinking_qt12.py`: Segments specific behavioral events based on coordinate-based triggers (e.g., proximity to feeders).

## 📂 Repository Structure

* `CRNN/`: Core directory for action recognition modules.
* `CRNN_train.py`: Main training script for the behavior classifier.
* `functions.py`: Custom PyTorch Dataloaders and model architecture definitions.


* `tracker.py`: Inference script combining detection, tracking, and movement quantification.
* `train.py`: Training pipeline for YOLO11 object detection.
* `video2image.py` / `video2img17.py`: Tools for video decoding and image calibration.
* `dataset.yaml`: Configuration for detection classes (e.g., `pig_lay`, `pig_stand`, `interactive`).

## 💻 Setup & Prerequisites

Dependencies are managed via `Pipfile`. It is recommended to use `pipenv` or install the following:

```bash
pip install torch torchvision ultralytics efficientnet_pytorch opencv-python pandas scikit-learn tqdm

```

## 📈 Training & Monitoring

* **To train the behavior classifier**:
```bash
python CRNN/CRNN_train.py

```


* **To train the object detector**:
```bash
python train.py

```


* **Monitoring**: The system integrates **TensorBoard**. Training loss and accuracy curves can be visualized by pointing TensorBoard to the `logs/` directory.
