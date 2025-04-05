#  Handwritten Digit Classification
## 📌 Project Overview

This project explores machine learning for image classification using a pixel-based dataset representing handwritten digits (0–9). The dataset contains grayscale pixel intensity values extracted from digit images along with their corresponding labels.

The goal is to:
- Preprocess the dataset
- Train a classification model
- Evaluate its performance
- Analyze per-digit accuracy

---

## 📊 Dataset

- **Source**: Custom dataset (similar to MNIST)
- **Columns**:
  - `label`: The actual digit (0–9)
  - `pixelX`: Grayscale intensity values (0–255) from selected pixels

---

## ❓ Question

**How accurately can a machine learning model classify digits (0–9) based on selected pixel values?**

---

## ✅ Answer

After building and evaluating the model, the results are:

- **Model Used**: `Logistic Regression` / `Random Forest` / `MLPClassifier` (choose your final one)
- **Overall Accuracy**: **`XX%`** *(Fill in after running the model)*
- **Per-Class Accuracy** (optional):
  - Digit 0: `__%`
  - Digit 1: `__%`
  - ...
- **Confusion Matrix**: Available in `results/accuracy_report.txt`

---
## 🛠️ Tools & Libraries

- Python (pandas, numpy)
- Scikit-learn
- Matplotlib / Seaborn (for visuals)
- Jupyter Notebook

