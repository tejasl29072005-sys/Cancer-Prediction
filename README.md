🧠 Breast Cancer Prediction using Machine Learning

A simple yet powerful machine learning project that predicts whether a tumor is **Malignant (Cancer)** or **Benign (No Cancer)** using patient diagnostic data.

📌 Project Overview

This project uses **Logistic Regression** and **Decision Tree Classifier** to classify breast cancer cases based on medical features. The dataset is preprocessed, scaled, and split into training and testing sets to evaluate model performance.

The goal is to compare different ML models and understand how well they perform in medical classification tasks.


📂 Dataset

* File used: `cancer.csv`
* Target column: `diagnosis`

  * `M` → Malignant (Cancer)
  * `B` → Benign (No Cancer)

### Data Preprocessing Steps:

* Converted diagnosis labels (M → 1, B → 0)
* Removed unnecessary columns (`id`, `Unnamed: 32`)
* Handled missing values
* Feature scaling using `StandardScaler`

---

## ⚙️ Workflow

1. Load dataset
2. Clean and preprocess data
3. Split into training and testing sets (80/20)
4. Scale features
5. Train models:

   * Logistic Regression
   * Decision Tree Classifier
6. Evaluate accuracy
7. Make sample predictions

---

## 🤖 Models Used

### 1. Logistic Regression

* Used for binary classification
* Outputs probability-based predictions

### 2. Decision Tree Classifier

* Works by splitting data into decision nodes
* Easy to interpret but may overfit

---

## 📊 Results

The model outputs accuracy for both classifiers:

```
Logistic Regression: ~High Accuracy (varies based on split)
Decision Tree      : ~High Accuracy (varies based on split)
```

It also shows **sample predictions** including:

* Actual label
* Predicted label
* Confidence score

---

## 🧪 Sample Output

```
Cancer Case:
  Actual    : Cancer
  Predicted : Cancer
  Confidence: 98.45%

No Cancer Case:
  Actual    : No Cancer
  Predicted : No Cancer
  Confidence: 95.30%
```

---

## 🛠️ Technologies Used

* Python 🐍
* Pandas
* Scikit-learn
* NumPy

---
 📈 Key Learnings

* Data preprocessing is critical in medical ML tasks
* Feature scaling improves model performance
* Logistic Regression works well for binary classification
* Decision Trees help in interpretability


 🚀 Future Improvements

* Add Random Forest / XGBoost for better accuracy
* Perform hyperparameter tuning
* Deploy model using Streamlit or Flask
* Add visualization dashboard

👨‍💻 Author

Developed by Tejas L



⭐ If you like this project

Feel free to star ⭐ the repository and explore more ML projects!
