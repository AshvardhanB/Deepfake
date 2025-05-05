
# 📘 Deepfake Detection using ResNet50 + LSTM

This project implements a deepfake detection system using a hybrid architecture that combines **ResNet50** (a powerful convolutional neural network) for spatial feature extraction and **LSTM** (Long Short-Term Memory) for temporal sequence modeling. The goal is to classify video sequences as either *Real* or *Fake*.

---

## 🔧 Architecture Overview

- **CNN Backbone**: ResNet50 (pretrained on ImageNet, without top layer)
- **Temporal Model**: LSTM
- **Input**: A sequence of 12 video frames, each resized to 160×160 pixels
- **Output**: Binary classification (Real = 0, Fake = 1)

---

## 📂 Dataset

- **Dataset Used**: Celeb-DF
- **Structure**:
  - Celeb-DF/
    - Celeb-real/
    - Celeb-synthesis/

Each video is converted into a sequence of frames, which are then cropped and resized for input to the model.

---

## 🛠️ Requirements

- TensorFlow / Keras
- OpenCV
- NumPy
- tqdm
- matplotlib
- seaborn
- scikit-learn

All required packages can be installed via pip.

---

## 🗂️ Project Workflow

1. **Import Dependencies**  
   Load necessary libraries including deep learning, preprocessing, visualization, and evaluation tools.

2. **Configuration Settings**  
   Define image size, sequence length, batch size, number of epochs, and dataset path.

3. **Face Detection**  
   Haar Cascade is used to detect and crop faces from each video frame. If no face is detected, the entire frame is resized.

4. **Frame Extraction**  
   Extract up to 12 frames from each video. If a video has fewer frames, it's padded to maintain sequence consistency.

5. **Feature Extraction with ResNet50**  
   Use ResNet50 (with top layers removed) to extract spatial features from individual frames using TimeDistributed layers.

6. **Sequence Learning with LSTM**  
   Frame-level features are passed to an LSTM layer to learn temporal relationships. A dropout and dense layer follow, leading to a binary sigmoid output.

7. **Dataset Preparation**  
   Paths and labels for real and fake videos are collected and split into training and testing sets.

8. **Custom Data Generator**  
   A custom Keras Sequence generator is used to efficiently load frame sequences and feed them to the model in batches.

9. **Training**  
   The model is compiled with binary crossentropy and trained with early stopping and class weighting.

10. **Model Saving**  
    After training, the model is saved to disk in Keras `.h5` format.

11. **Evaluation**  
    The model is evaluated using accuracy, ROC AUC, Cohen's Kappa, recall score, F1 scores, and confusion matrix.

12. **Statistical Testing**  
    Advanced testing like Wilcoxon signed-rank test and McNemar’s test is used to compare this model’s performance with others (e.g., MobileNetV2 + LSTM).

---

## 📈 Results

- **Accuracy**: ~79%
- **ROC AUC**: 0.72
- **F1 Score (Fake)**: 0.81
- **F1 Score (Real)**: 0.65

---

## 🔬 Statistical Analysis

- **Wilcoxon Test**: Statistical difference observed between ResNet50+LSTM and other models (e.g., MobileNetV2+LSTM)
- **McNemar's Test**: Significant misclassification difference, confirming comparative performance

---

## 🎥 Custom Video Prediction

This model can be used to analyze any custom `.mp4` video. A real-world video is passed through the same preprocessing pipeline (face detection, frame extraction, normalization) and evaluated using the trained model. The output is either *Fake* or *Real*.

---

## ⚠️ Limitations

- Lower accuracy compared to MobileNetV2+LSTM in some scenarios
- Performance drops on very subtle manipulations or high compression artifacts
- Higher computational cost due to ResNet50’s complexity

---

## 🌱 Future Work

- Integrate ensemble predictions with other models like MobileNetV2+LSTM
- Explore use of attention mechanisms and 3D CNNs
- Enhance deployment by converting to ONNX or TensorFlow Lite
- Support streaming input for real-time detection
