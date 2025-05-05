# 📘 Deepfake Detection using MobileNetV2 + LSTM

This project implements a deep learning pipeline to detect deepfake videos using a hybrid CNN-RNN architecture. It combines MobileNetV2 for spatial feature extraction and LSTM for temporal sequence learning.

---

## 🔧 Architecture Overview

- CNN Backbone: MobileNetV2 (pretrained on ImageNet)
- Temporal Model: LSTM
- Input: 12-frame sequences (160x160 each)
- Output: Binary classification (Real or Fake)

---

## 📂 Dataset

- Source: Celeb-DF (v1 or v2)
- Structure:
  - Celeb-DF/
    - Celeb-real/
    - Celeb-synthesis/

Each video is split into frames and preprocessed.

---

## 🛠️ Requirements

You will need:
- TensorFlow
- OpenCV
- NumPy
- tqdm
- matplotlib
- seaborn
- scikit-learn

Install them using pip.

---

## 🗂️ Project Workflow

1. **Import Dependencies**  
   Load all required libraries including Keras, TensorFlow, OpenCV, etc.

2. **Configuration**  
   Define constants like frame size, sequence length, batch size, epochs, and dataset path.

3. **Face Detection**  
   Faces are detected from each video frame using Haar cascades. If no face is found, the entire frame is resized.

4. **Frame Extraction**  
   Each video is converted into a fixed-length sequence of frames. If the video is too short, it's padded with black frames.

5. **Feature Extraction with CNN**  
   MobileNetV2 is used to extract spatial features from each frame using TimeDistributed layer.

6. **LSTM Classifier**  
   The frame-wise features are passed through an LSTM layer to learn temporal patterns. Final output is a single sigmoid-activated unit for binary classification.

7. **Dataset Loader**  
   Paths to real and fake videos are loaded with appropriate labels (0 for real, 1 for fake).

8. **Custom Data Generator**  
   A Keras Sequence generator is used to efficiently feed frame sequences and labels to the model during training.

9. **Training**  
   The model is trained with class weights and early stopping. Accuracy and loss are monitored on a validation set.

10. **Model Saving**  
    The trained model is saved in Keras format.

11. **Evaluation**  
    Model is evaluated on a test set and metrics like accuracy, F1 score, confusion matrix, and AUC are computed.

12. **Statistical Testing**  
    Wilcoxon and McNemar’s tests are used to statistically compare the model with other architectures like ResNet50.

---

## 📈 Results

- Accuracy: 90%
- AUC Score: 0.96
- F1 Score (Fake): 0.93
- F1 Score (Real): 0.85

---

## 🔬 Statistical Significance

- Wilcoxon Test: p = 0.0069
- McNemar's Test: p < 0.0001  
These results indicate a statistically significant improvement over baseline models.

---

## 🎥 Custom Video Prediction

After training, you can use your model to predict whether a new video is real or fake. This is done by extracting frames, preprocessing them like training data, and running inference using the trained model.

The prediction will print either "Fake" or "Real" based on the model’s output probability.

---

## ⚠️ Limitations

- Dataset limited to Celeb-DF only.
- May struggle on unseen deepfake generation methods.
- Batch size and sequence length constrained due to hardware.

---

## 🌱 Future Work

- Incorporate other datasets like DFDC or FaceForensics++.
- Try transformer-based or attention-based temporal models.
- Convert to TensorFlow Lite for mobile deployment.
- Build a lightweight web app for public use.
