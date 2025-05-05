# 📊 Comparison of MobileNetV2+LSTM vs. ResNet50+LSTM for Deepfake Detection

This notebook analyzes and compares the performance of two hybrid deep learning models—**MobileNetV2+LSTM** and **ResNet50+LSTM**—for deepfake video classification.

---

## 🎯 Objective

To statistically evaluate which model performs better by comparing predicted probabilities and classifications across the same test dataset.

---

## 📈 Models Compared

- **Model 1**: MobileNetV2 + LSTM  
- **Model 2**: ResNet50 + LSTM

Both models were trained on the **Celeb-DF** dataset using the same number of video frame sequences (12 frames per sample), identical preprocessing, and evaluation metrics.

---

## 🧪 Metrics Evaluated

The comparison includes:
- Predicted class probabilities (for AUC and statistical tests)
- Final predicted labels (for accuracy and confusion matrix)
- F1 Scores (Fake vs. Real)
- ROC AUC Scores

---

## 🔬 Statistical Tests Used

### ✅ Wilcoxon Signed-Rank Test

- Purpose: Compare the predicted probabilities of both models
- Result:
  - Indicates whether one model significantly outperforms the other
  - Null hypothesis: Both models have similar distributions of probability outputs

### ✅ McNemar’s Test

- Purpose: Compare misclassification patterns
- Result:
  - Indicates whether there's a statistically significant difference in error patterns between models
  - Useful when both models make different types of mistakes on the same examples

---

## 📊 Results Summary

| Metric            | MobileNetV2+LSTM | ResNet50+LSTM |
|-------------------|------------------|----------------|
| Accuracy          | 90%              | 79%            |
| AUC Score         | 0.96             | 0.72           |
| F1 Score (Fake)   | 0.93             | 0.81           |
| F1 Score (Real)   | 0.85             | 0.65           |
| Wilcoxon p-value  | 0.0069           | —              |
| McNemar p-value   | < 0.0001         | —              |

**Conclusion**: MobileNetV2+LSTM statistically outperforms ResNet50+LSTM in this setting.

---

## ⚠️ Notes

- Dataset used for evaluation: same test set used in both models
- Probabilities and predictions loaded from `.npy` files
- Results depend on the specific train-test split, random seed, and training convergence

---

## 📌 Conclusion

The statistical evidence suggests that the MobileNetV2+LSTM model is **significantly better** than the ResNet50+LSTM model for the deepfake detection task under the same experimental conditions.
