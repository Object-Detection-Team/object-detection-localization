# Project Directory Scheme for Google Colab

This document outlines the directory structure for your multi-model object detection project, specifically adapted for training on **Google Colab**.

## Google Drive Setup

To run the training notebooks successfully, you must mirror your local project structure to your Google Drive.

1.  **Upload Datasets**: Upload the **entire** `labeled-datasets` folder to the **root** of your Google Drive (`MyDrive`).
    *   Path on Drive should be: `/content/drive/MyDrive/labeled-datasets/`

## Directory Tree (Google Drive View)

```
/content/drive/MyDrive/
└── labeled-datasets/
    ├── Fruit classification.v1i.yolov8/
    │   ├── data_local.yaml   # Config used by Colab
    │   ├── train/
    │   └── ...
    ├── Fruits by YOLO.v1i.yolov5pytorch/
    │   ├── data_local.yaml   # Config used by Colab
    │   ├── train/
    │   └── ...
    ├── Fruits by YOLO.v1-fruits-detection.yolov11/
    │   ├── data_local.yaml   # Config used by Colab
    │   ├── train/
    │   └── ...
    └── training_results/     # [NEW] Where results are saved
        ├── yolov5_fruits/
        ├── yolov8_fruits/
        └── yolov11_fruits/
```

## Notebooks

The notebooks in the `notebooks/` folder are now configured to:
1.  **Mount Google Drive** automatically.
2.  Install `ultralytics`.
3.  Point to the correct dataset path on Drive (`/content/drive/MyDrive/labeled-datasets/...`).
4.  **Save Results**: Automatically copy the best weights and training graphs to `/content/drive/MyDrive/labeled-datasets/training_results/<model_name>`.

## Running the Training

1.  Open [Google Colab](https://colab.research.google.com/).
2.  Upload one of the notebooks from the `notebooks/` folder (e.g., `train_yolov5.ipynb`).
3.  Ensure your `labeled-datasets` folder is in your Google Drive root.
4.  Run all cells.

## Model Versions

- **YOLOv8**: `yolov8m.pt`
- **YOLOv5**: `yolov5mu.pt`
- **YOLO11**: `yolo11m.pt`
