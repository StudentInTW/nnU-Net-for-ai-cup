# 🏆 nnU-Net for AI CUP: [Insert Competition Name, e.g., Aortic Valve Segmentation]

This repository contains the official implementation and solution pipeline for the **[Year] AI CUP Competition**. Leveraging **nnU-Net**, the state-of-the-art framework for biomedical image segmentation, this project encapsulates a complete end-to-end pipeline—from custom data preprocessing and self-configuring model training to advanced heuristic post-processing and inference.

## ✨ Key Features & Technical Highlights

* **2.5D Dimension Bridging Architecture**: 
  Designed a custom data pipeline to seamlessly encapsulate 2D image slices and YOLO-format bounding box annotations into the 3D NIfTI format required by the nnU-Net engine. Post-inference, a precision `slice_map` mechanism is utilized to project the 3D predictions accurately back into the original 2D coordinate system.
* **Math-Driven Heuristic Post-Processing**:
  * **Noise Suppression**: Implemented a connected-component analysis with a dynamic volume filter (`MIN_PIXEL_AREA`) to effectively eliminate false-positive predictions and artifacts.
  * **Pseudo-Confidence Score Derivation**: Addressed the inherent lack of bounding-box confidence scores in semantic segmentation models by introducing a non-linear mapping formula: $0.5 + 0.5 \times \tanh(\text{Area} / 1000.0)$. This transforms the predicted mask area into a robust Confidence Score for object detection evaluation.
* **Cloud Compute & I/O Optimization**:
  Engineered a smart caching mechanism that optimizes heavy `.npz` preprocessed files into `.npy` memory-mapped (`mmap`) formats. Combined with automated cloud backups, this drastically reduces I/O bottlenecks, memory overhead, and computational costs during server restarts (e.g., Google Colab environments).

## 📂 Project Architecture

To comply with nnU-Net's strict environment variable requirements, the project follows this directory structure:

```text
├── nnu-net_for_ai_cup.ipynb    # Main pipeline (Data conversion, Training, Inference)
├── nnUNet_raw/                 # Converted Task datasets (e.g., Task502_AorticValve)
├── nnUNet_preprocessed/        # Auto-generated fingerprints and optimized .npy files
├── nnUNet_results/             # Model weights (.pth) and validation outputs
└── nnUNet_infer_final/         # Final 3D NIfTI prediction outputs
```

## 🚀 Quick Start

### 1. Environment Setup
It is highly recommended to run this pipeline in a Linux environment with GPU acceleration (or Google Colab / Kaggle).

```bash
# Create a virtual environment
conda create -n nnunet_env python=3.9 -y
conda activate nnunet_env

# Install PyTorch (Adjust according to your CUDA version)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install nnU-Net v2
pip install nnunetv2
```

### 2. Execution Pipeline
The complete workflow is documented within `nnu-net_for_ai_cup.ipynb`. The core steps are:

1. **Dataset Formatting**: Convert raw 2D images and YOLO text annotations into nnU-Net compatible JSON and NIfTI (`.nii.gz`) formats.
2. **Planning & Preprocessing**: Extract dataset fingerprints and optimize data structures.
   ```bash
   nnUNetv2_plan_and_preprocess -d [Task_ID] --verify_dataset_integrity
   ```
3. **Model Training**: Execute 5-fold cross-validation.
   ```bash
   nnUNetv2_train [Task_ID] 2d 0  # Trains Fold 0 for the 2D configuration
   ```
4. **Inference & Post-Processing**: Predict on the testing set and convert the output masks back to bounded boxes with confidence scores.
   ```bash
   nnUNetv2_predict -i [TEST_IMG_DIR] -o [INFER_DIR] -d [Task_ID] -c 2d -f all
   ```

## 📊 Competition Results

* **Evaluation Metric**: [e.g., Mean Average Precision (mAP) / Dice Score]
* **Public Leaderboard Score**: `0.84`

> **Developer Note:** During the implementation, one of the main challenges was dealing with [Mention a specific challenge, e.g., severe class imbalance or blurry lesion boundaries]. By implementing [Mention your solution, e.g., the heuristic area filter and TTA], the model's robustness and precision on the unseen test set improved significantly.
```
