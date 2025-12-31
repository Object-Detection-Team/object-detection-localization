# Comparative Analysis of YOLOv5, YOLOv8, and YOLOv11 Architectures for Real-Time Fruit Detection

**Ramazan Yildiz**, **Abdelrahman Mohamed**, **Beyza Güler**

Department of Computer Engineering, Ataturk University, Erzurum, Turkey

---

## Abstract

This paper presents a comparative analysis of YOLOv5, YOLOv8, and YOLOv11 Medium architectures for real-time fruit detection, trained on the Roboflow "Fruits by YOLO" dataset containing 2,697 training and 187 validation images across nine classes (Apple, Banana, Grapes, Kiwi, Mango, Orange, Pineapple, Sugerapple, Watermelon). All models were trained for 50 epochs on Google Colab Pro using NVIDIA Tesla T4 GPUs with identical hyperparameters (640×640 resolution, batch size 16, AdamW optimizer). YOLOv11 Medium achieved the highest accuracy with 77.8% mAP@50 and 75.5% recall, while YOLOv5 Medium demonstrated 23% faster inference at 10.4 ms per image compared to 12.8 ms for YOLOv11. We deployed a Streamlit web application featuring WebRTC-based real-time video processing at 30 FPS, dynamic model switching, and thread-safe adjustable confidence/IoU thresholds through a dark-themed interface. Our findings indicate that YOLOv11 is optimal for accuracy-critical applications while YOLOv5 remains preferable for latency-sensitive edge deployments.

**Keywords:** YOLOv5, YOLOv8, YOLOv11, Fruit Detection, Streamlit, WebRTC, Real-Time Object Detection

---

## I. Introduction

This research investigates the performance differences between three YOLO architecture generations applied to fruit detection using the Roboflow "Fruits by YOLO" dataset. The study addresses a practical question faced by computer vision practitioners: given the rapid evolution from YOLOv5 (2020) through YOLOv8 (2023) to YOLOv11 (2024), which architecture provides optimal performance for real-time agricultural object detection applications?

Our investigation focuses on detecting nine specific fruit classes—Apple, Banana, Grapes, Kiwi, Mango, Orange, Pineapple, Sugerapple, and Watermelon—representing diverse visual characteristics including varying colors (yellow bananas, green kiwis, red apples), shapes (elongated bananas, spherical oranges, irregular grapes clusters), and textures (smooth apple skin, textured pineapple exterior). This diversity provides a rigorous testbed for evaluating detection robustness across visually distinct object categories.

Beyond comparative model analysis, we developed a production-ready Streamlit web application enabling practitioners to deploy and compare all three trained models within a unified interface. The application leverages WebRTC for low-latency webcam streaming, processes frames through PyAV for efficient video handling, and provides real-time inference visualization at approximately 30 frames per second. Users can dynamically switch between YOLOv5 (optimized for speed), YOLOv8 (balanced performance), and YOLOv11 (maximum accuracy) without application restart, and adjust inference parameters including confidence threshold (0.00-1.00) and IoU threshold for non-maximum suppression through an intuitive sidebar interface.

---

## II. Related Work

The YOLO architecture family has undergone substantial evolution since its introduction. YOLOv5, released in 2020 by Ultralytics, introduced a PyTorch-native implementation with automatic anchor computation and multiple model size variants (nano through extra-large). YOLOv8, released in January 2023, represented a significant architectural revision with anchor-free detection heads, decoupled classification and regression branches, and the improved C2f backbone module replacing the earlier CSP blocks. YOLOv11, released in late 2024, further refined the C2f architecture and introduced enhanced multi-scale feature aggregation, claiming state-of-the-art performance on COCO benchmarks.

Within agricultural applications, object detection models have been applied to crop monitoring, yield estimation, and automated harvesting systems. Fruit detection specifically has relevance to precision agriculture, automated sorting facilities, and cashierless retail checkout systems. However, systematic comparisons of YOLOv5, YOLOv8, and YOLOv11 under controlled training conditions on identical fruit datasets remain limited in published literature.

---

## III. Methodology

### A. Dataset

We utilized the "Fruits by YOLO" dataset obtained from the Roboflow platform, specifically the YOLOv11-formatted version for consistency. The dataset comprises 2,884 total images partitioned into 2,697 training images and 187 validation images. All annotations follow YOLO format with normalized bounding box coordinates (center_x, center_y, width, height) and integer class indices.

The nine fruit classes and their visual characteristics are summarized in Table I. Classes were selected to provide diversity in color profiles (ranging from yellow Bananas to green Kiwis to red Apples), geometric shapes (spherical Oranges versus elongated Bananas versus clustered Grapes), and surface textures (smooth Apple skin versus rough Pineapple exterior versus fuzzy Kiwi surface).

**Table I: Fruit Classes and Visual Characteristics**

| Class | Dominant Color | Shape | Distinguishing Features |
|-------|---------------|-------|------------------------|
| Apple | Red/Green | Spherical | Smooth skin, stem indent |
| Banana | Yellow | Elongated | Curved shape, clustered presentation |
| Grapes | Purple/Green | Clustered | Small individual berries in bunches |
| Kiwi | Brown/Green | Oval | Fuzzy brown exterior |
| Mango | Yellow/Orange | Oval | Gradient coloring |
| Orange | Orange | Spherical | Textured peel, uniform color |
| Pineapple | Yellow/Brown | Cylindrical | Crown leaves, diamond pattern |
| Sugerapple | Green | Irregular | Bumpy segmented surface |
| Watermelon | Green/Red | Large oval | Striped exterior, large size |

### B. Training Configuration

All training was conducted on Google Colab Pro using NVIDIA Tesla T4 GPUs with 16 GB video memory. We selected Medium model variants (YOLOv5m, YOLOv8m, YOLOv11m) to ensure comparable parameter counts across architectures. Training hyperparameters were held constant across all experiments to ensure fair comparison, as detailed in Table II.

**Table II: Training Hyperparameters**

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Image Size | 640 × 640 pixels |
| Batch Size | 16 |
| Optimizer | AdamW |
| Training Images | 2,697 |
| Validation Images | 187 |
| GPU | NVIDIA Tesla T4 (16 GB VRAM) |
| Platform | Google Colab Pro |

Three separate Jupyter notebooks were developed for each architecture: train_yolov5.ipynb, train_yolov8.ipynb, and train_yolov11.ipynb. Each notebook mounts Google Drive for persistent storage, installs the Ultralytics library, configures dataset paths pointing to the Roboflow data on Drive, executes training with specified hyperparameters, and exports trained weights (best.pt), confusion matrices, and training curve visualizations to designated output directories.

### C. Evaluation Metrics

Model performance was assessed using four metrics: mean Average Precision at IoU threshold 0.50 (mAP@50) as the primary accuracy measure averaging precision across all nine fruit classes; Recall measuring the proportion of actual fruit instances successfully detected; Precision quantifying resistance to false positive detections; and Inference Speed measured as average milliseconds per image on T4 hardware.

---

## IV. System Implementation

### A. Web Application Architecture

The detection system was implemented as a Streamlit web application featuring a dark theme with gold accent colors. The interface provides three main pages: Overview for project information and model comparison, Live Detection for real-time webcam inference, and Analytics Dashboard for detection statistics. The application supports dynamic switching between YOLOv5, YOLOv8, and YOLOv11 trained weights, utilizing Streamlit's caching mechanism to minimize model loading latency during runtime.

### B. WebRTC Video Processing

Real-time video capture and processing utilizes the streamlit-webrtc library with custom video processor implementation. The YOLOv8VideoProcessor class extends VideoProcessorBase and implements the recv() method for frame-by-frame inference. Each incoming WebRTC frame is converted to BGR24 numpy array format via PyAV, processed through the active YOLO model with user-specified confidence and IoU thresholds, annotated with detection bounding boxes and class labels using the Ultralytics plot() method, and returned as an annotated VideoFrame for display.

WebRTC configuration includes STUN servers (stun.l.google.com:19302 through stun4.l.google.com:19302) for NAT traversal, enabling the application to function across network configurations. Processing achieves approximately 30 FPS with 480×480 inference resolution optimized for real-time performance.

### C. Thread-Safe Parameter Adjustment

The InferenceSettings dataclass implements thread-safe parameter management between the Streamlit UI thread and the WebRTC video processing thread. A threading.Lock protects concurrent access to confidence and IoU threshold values. The update() method safely writes new parameter values from UI slider changes, while the snapshot() method safely reads current values for each inference call. This design enables users to adjust detection sensitivity in real-time without stream interruption or race conditions.

Sidebar sliders allow adjustment of Confidence Threshold (0.00-1.00, default 0.05) controlling minimum detection confidence, and IoU Threshold (0.00-1.00, default 0.45) controlling non-maximum suppression aggressiveness for overlapping detections.

---

## V. Results and Discussion

### A. Quantitative Results

The experimental evaluation yielded the performance metrics summarized in Table III. YOLOv11 Medium achieved the highest mAP@50 of 77.8% and recall of 75.5%, representing improvements of 1.3 and 3.4 percentage points respectively over YOLOv5 Medium. However, YOLOv5 Medium demonstrated 23% faster inference at 10.4 ms per image compared to 12.8 ms for YOLOv11, and achieved the highest precision of 82.3%, indicating fewer false positive detections.

**Table III: Comparative Performance Results**

| Model | mAP@50 | Recall | Precision | Inference Speed | Training Time |
|-------|--------|--------|-----------|-----------------|---------------|
| YOLOv11 Medium | 77.8% | 75.5% | 81.2% | 12.8 ms | 1.49 hours |
| YOLOv8 Medium | 76.5% | 74.8% | 80.5% | 11.6 ms | 1.72 hours |
| YOLOv5 Medium | 76.7% | 72.1% | 82.3% | 10.4 ms | 1.25 hours |

YOLOv8 Medium positioned between the other two variants in most metrics, achieving 76.5% mAP@50 with 11.6 ms inference speed. Training duration varied from 1.25 hours for YOLOv5 to 1.72 hours for YOLOv8, with YOLOv11 requiring 1.49 hours—suggesting that YOLOv11's architectural refinements enabled faster convergence than YOLOv8 despite similar complexity.

### B. Class-Level Analysis

Confusion matrix analysis revealed that visually distinctive classes achieved highest detection rates across all models. Banana (yellow, curved shape), Watermelon (large size, striped pattern), and Pineapple (crown leaves, diamond texture) consistently exceeded 90% accuracy. Moderate confusion occurred between Apple and Orange due to similar spherical shapes and overlapping color ranges in certain ripeness states. Kiwi and Mango showed occasional mutual misclassification attributable to comparable oval shapes and brown-green coloration.

The nine-class detection task presented sufficient complexity to differentiate model capabilities while remaining tractable for the dataset size. Sugerapple, being less common in training data and having irregular bumpy surface geometry, showed the highest variability in detection accuracy across models.

### C. Deployment Performance

The Streamlit application successfully achieves real-time performance targets on consumer hardware. With YOLOv5 weights active, the system maintains consistent 30 FPS processing on laptop webcams. YOLOv11 weights reduce effective framerate to approximately 25 FPS due to increased inference latency, though this remains acceptable for interactive demonstration purposes.

The model switching functionality operates seamlessly, with Streamlit's caching mechanism ensuring that switching from YOLOv5 to YOLOv11 incurs only a brief loading delay on first selection, with subsequent switches occurring instantaneously from cache. The dark theme with gold accent design provides professional presentation suitable for academic demonstration and potential commercial deployment.

---

## VI. Limitations

The dataset size of approximately 3,000 images, while sufficient for demonstrating comparative trends, remains modest compared to industrial-scale training sets. The Roboflow "Fruits by YOLO" dataset may not fully represent real-world variability in lighting conditions, occlusion patterns, and produce presentation contexts encountered in commercial sorting or retail applications.

Training was limited to 50 epochs due to Google Colab session constraints and computational budget considerations. Extended training (100-200 epochs) with learning rate scheduling might yield different comparative outcomes, particularly for architectures that benefit from longer convergence periods. The inference speed measurements reflect T4 GPU performance and may not directly translate to other deployment targets such as edge devices or alternative GPU architectures.

Hardware detection in the Streamlit application currently defaults to CPU on systems without CUDA or Apple Silicon MPS support, which significantly impacts inference speed. The WebRTC implementation, while functional, may encounter connectivity issues in enterprise network environments with restrictive firewall configurations.

---

## VII. Conclusion

This study systematically compared YOLOv5, YOLOv8, and YOLOv11 Medium architectures trained on the Roboflow "Fruits by YOLO" dataset containing nine fruit classes and 2,697 training images. YOLOv11 Medium achieved the highest detection accuracy with 77.8% mAP@50 and 75.5% recall on the validation set of 187 images. YOLOv5 Medium demonstrated 23% faster inference at 10.4 ms per image while sacrificing only 1.1 percentage points in mAP@50, making it preferable for latency-critical edge deployments.

The deployed Streamlit application provides a functional platform for real-time fruit detection, featuring WebRTC-based webcam streaming at 30 FPS, dynamic switching between all three trained model variants, and adjustable confidence and IoU thresholds. The professional dark-themed interface demonstrates production-ready design suitable for academic presentation and potential commercial applications in agricultural automation or retail checkout systems.

---

## VIII. Future Work

Several promising research directions emerge from this work. Extended training on Google Colab Pro+ or local GPU infrastructure could enable 100+ epoch training schedules with cosine annealing learning rate policies. Model optimization through INT8 quantization and structured pruning would reduce computational requirements for deployment on embedded systems such as NVIDIA Jetson Nano or Raspberry Pi with neural compute sticks.

Integration of Convolutional Block Attention Module (CBAM) into the YOLOv11 backbone presents a particularly promising enhancement for our fruit detection task. CBAM's channel attention mechanism could help the model learn which feature channels are most discriminative for distinguishing between visually similar fruits like Apple and Orange, while spatial attention could improve localization accuracy for irregularly-shaped classes like Grapes clusters and Sugerapple. Based on published benchmarks, CBAM integration could potentially improve our current 77.8% mAP@50 by an additional 1-3 percentage points with minimal inference speed degradation.

Equally important for practical deployment is the integration of Explainable Artificial Intelligence (XAI) techniques. Implementing Gradient-weighted Class Activation Mapping (Grad-CAM) would generate visual heatmaps showing which image regions—such as the distinctive crown of Pineapple or the striped pattern of Watermelon—most influence detection decisions. This transparency is essential for quality control applications where operators must understand and trust automated sorting decisions. Complementarily, SHapley Additive exPlanations (SHAP) analysis could quantify feature attributions, revealing whether detections rely on expected visual characteristics (color, shape, texture) versus spurious dataset biases. Such interpretability enhancements would significantly increase confidence in deploying our fruit detection system for real-world agricultural quality inspection and automated retail checkout applications.

---

## Acknowledgments

The authors acknowledge Google Colab for providing computational resources and Roboflow for hosting the "Fruits by YOLO" dataset. This work was conducted for the Introduction to Artificial Intelligence course at Ataturk University, Department of Computer Engineering.

---

## References

[1] G. Jocher, "YOLOv5 by Ultralytics," GitHub repository, 2020. [Online]. Available: https://github.com/ultralytics/yolov5

[2] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," GitHub repository, 2023. [Online]. Available: https://github.com/ultralytics/ultralytics

[3] Ultralytics, "YOLO11 Documentation," 2024. [Online]. Available: https://docs.ultralytics.com/models/yolo11/

[4] Roboflow, "Fruits by YOLO Dataset," 2024. [Online]. Available: https://roboflow.com

[5] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional Block Attention Module," in *Proceedings of the European Conference on Computer Vision (ECCV)*, 2018, pp. 3-19.

[6] R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," in *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2017, pp. 618-626.

[7] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017, pp. 4765-4774.

[8] Streamlit Inc., "Streamlit Documentation," 2024. [Online]. Available: https://docs.streamlit.io
