<div align="center">
  <!-- You can replace this link with a GIF or image of volleyball bounding boxes later -->
  <img src="https://github.com/user-attachments/assets/22cc8c54-f3c7-4900-a9db-3e37fffac5ad" alt="Background Image" width="95%" />
</div>

<h1 align="center">Hierarchical Deep Temporal Model for Group Activity Recognition</h1>

<p align="center">
  A modern, highly modular PyTorch implementation of the <strong>CVPR 2016 paper</strong>: <a href="https://arxiv.org/pdf/1607.02643"><em>A Hierarchical Deep Temporal Model for Group Activity Recognition</em></a>.  
  This project captures both individual player actions and group-level temporal dynamics using a Two-Stage ResNet50 + LSTM architecture.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python">
  <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white" alt="Kaggle">
</p>

## Table of Contents
1. [Key Features & Architecture Updates](#key-features--architecture-updates)
2. [Project Structure](#project-structure)
3. [Dataset Overview](#dataset-overview)
4. [Ablation Study (The Baselines)](#ablation-study-the-baselines)
5. [Getting Started (Usage)](#getting-started-usage)
6. [Cloud Training (Kaggle/Colab Bridge)](#cloud-training-kagglecolab-bridge)

---

## Key Features & Architecture Updates
This repository upgrades the original 2016 Caffe implementation to modern standards, focusing on scalability and performance:

* **Modern Backbone:** Replaced the original AlexNet with a `ResNet50` feature extractor.
* **Automatic Mixed Precision (AMP):** Integrated PyTorch `GradScaler` and `autocast` to halve VRAM usage and double training speed on modern GPUs.
* **Modular Configuration:** All hyperparameter tuning, model sizing, and data paths are controlled entirely via `.yaml` configuration files. 
* **Seamless Cloud-to-Local CI/CD:** Built-in environment detection (`env_utils.py`) automatically routes dataset paths and handles multiprocessing deadlocks (`spawn` vs `fork`) depending on whether the code is running locally or on Kaggle/Colab.
* **Rich Logging:** Replaced standard terminal outputs with a custom `logging` pipeline that outputs timestamped, batch-level metrics to text files alongside TensorBoard visualizations.

---

## Project Structure
```text
Hierarchical-Deep-Temporal-Model-for-Group-Activity-Recognition/
├── configs/                  # YAML files controlling all model/training parameters
│   ├── baseline_1.yaml
│   ├── baseline_3_phase_A.yaml
│   ├── baseline_3_phase_B.yaml
│   └── baseline_4.yaml
├── data_utilities/           # Data ingestion, Pickling, and PyTorch Datasets
│   ├── box_annot.py
│   ├── data_annot_loader.py
│   └── dataset.py            # Contains GroupActivity, PersonAction, and Sequence Datasets
├── loader_utils/             # Core engineering utilities
│   ├── env_utils.py          # Auto-detects Kaggle vs Local environments
│   └── helper.py             # Config parsers, seed setting, and formatting loggers
├── models/                   # Architecture and Training scripts
│   ├── train_utils.py        # Universal training/validation loop with AMP and TensorBoard
│   ├── baseline_1/           # Standard ResNet50 Image Classifier
│   ├── baseline_3/           # Spatial Person & Group Classifier
│   └── baseline_4/           # Long-term Recurrent Convolutional Network (LRCN)
└── requirements.txt
```

---

## Dataset Overview
The dataset utilizes publicly available YouTube volleyball videos, containing 4,830 annotated frames from 55 videos. 

#### Group Activity Labels
| Group Activity Class | Instances |
|-----------------------|-----------|
| Right set            | 644       |
| Right spike          | 623       |
| Right pass           | 801       |
| Right winpoint       | 295       |
| Left winpoint        | 367       |
| Left pass            | 826       |
| Left spike           | 642       |
| Left set             | 633       |

#### Player Action Labels
| Action Class | Instances |
|--------------|-----------|
| Waiting      | 3,601     |
| Setting      | 1,332     |
| Digging      | 2,333     |
| Falling      | 1,241     |
| Spiking      | 1,216     |
| Blocking     | 2,458     |
| Jumping      | 341       |
| Moving       | 5,121     |
| Standing     | 38,696    |


### Dataset Organization

- **Videos**: 55, each assigned a unique ID (0–54).
- **Train Videos**: 1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54.
- **Validation Videos**: 0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51.
- **Test Videos**: 4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47.



For further information about dataset, you can check out the paper author's repository: [link](https://github.com/mostafa-saad/deep-activity-rec)

---
## Ablation Study (The Baselines)
This project breaks down the paper's architecture into distinct baselines to analyze the impact of spatial vs. temporal features:

* **Baseline 1 (Image Classification):** A purely spatial model using ResNet50 to classify the group activity from a single, static video frame.
* **Baseline 3 (Fine-tuned Person Classification):** Deploys ResNet50 to extract 2048-d features from individual player bounding boxes. Features are pooled across all players in a frame and fed to a classifier.
* **Baseline 4 (LRCN - Temporal Image Features):** Introduces time. A sequence of 9 frames (a clip) is passed through ResNet50 to extract spatial features, which are chronologically fed into an `LSTM` to understand group motion before classification.

---

## Getting Started (Usage)

### 1. Clone & Install
```bash
git clone [https://github.com/Amr2054/Hierarchical-Deep-Temporal-Model-for-Group-Activity-Recognition.git)
cd Hierarchical-Deep-Temporal-Model-for-Group-Activity-Recognition
```

### 2. Dataset Preparation
Ensure the Volleyball dataset is downloaded. Parse the raw text annotations into the optimized `.pkl` format:
```bash
python -m data_utilities.data_annot_loader
```

### 3. Train a Model
Because the project uses module execution, run training scripts from the root directory and pass the corresponding YAML config:
```bash
# Example: Training the Baseline 4 LRCN Model
python -m models.baseline_4.train --config configs/baseline_4.yaml
```

Outputs, `.pth` weights, TensorBoard logs, and Confusion Matrices will automatically save to `models/baseline_X/outputs/run_[timestamp]/`.

---

## Cloud Training (Kaggle/Colab Bridge)
This repository is designed to be edited in a local IDE (like PyCharm/VSCode) but executed on cloud GPUs without path errors.

1. Push your code to GitHub.
2. In a Kaggle Notebook, clone your private repo using a Personal Access Token (PAT).
3. The internal `setup_environment()` function will automatically detect the Kaggle kernel, reroute the dataset paths to `/kaggle/input/`, write outputs to `/kaggle/working/`, and configure the DataLoader `num_workers` to prevent container deadlocks. 
```bash
!git pull origin main
!python -m models.baseline_4.train --config configs/baseline_4.yaml
```

