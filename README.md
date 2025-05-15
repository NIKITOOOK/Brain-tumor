# 🧠 Brain Tumor Detection using Deep Learning

This project focuses on the automatic detection of brain tumors using state-of-the-art deep learning architectures such as **EfficientNet**, **ResNet**, and **YOLOv5n**. The dataset used is publicly available on Kaggle: [Brain Tumor Detection](https://www.kaggle.com/datasets/ultralytics/brain-tumor/data).

---

## 📁 Dataset Overview

The dataset includes MRI images categorized into four classes:

- `glioma_tumor`
- `meningioma_tumor`
- `pituitary_tumor`
- `no_tumor`

**Dataset Details:**

- ~3,264 images
- Format: JPG
- Varying dimensions (resized to 224x224 or 640x640 depending on model)
- Organized into subfolders by class

---

## 🛠️ Models Used

### `EfficientNetB0_ResNet50.ipynb`
- Image classification using:
  - EfficientNetB0
  - ResNet50
- Framework: `Keras`, `TensorFlow`
- Data augmentation using `ImageDataGenerator`
- Fine-tuning on pre-trained weights

### `brain_tumor.ipynb`
- A simple custom CNN for classification
- Manual train/test split
- Emphasis on simplicity and interpretability

### `yolo11n__brain_tumor_.ipynb`
- Object detection and localization using `YOLOv5n`
- Framework: `Ultralytics YOLOv5` on `PyTorch`
- Outputs bounding box coordinates and class labels

---

## ⚙️ Installation

```bash
git clone https://github.com/your_username/brain-tumor-detection.git
cd brain-tumor-detection
pip install -r requirements.txt
```


📌 Dependencies
Main dependencies for the project:
* Python 3.8+
* TensorFlow >= 2.8
* Keras
* PyTorch
* Ultralytics (pip install ultralytics)
* OpenCV
* NumPy
* Matplotlib

🚀 How to Run
1. Download the dataset from Kaggle
2. Extract it into ./data/brain_tumor
3. Launch any of the .ipynb notebooks (via Jupyter or Google Colab)
4. Follow the notebook cells to train and evaluate models

📈 Possible Improvements
Semantic segmentation with U-Net or DeepLabV3,
Hyperparameter optimization (LR, optimizers, schedulers),
Model ensembling,
Using Vision Transformers (ViT),
Precise annotation for YOLO (custom bounding boxes).

👨‍💻 Author
[https://github.com/NIKITOOOK]
