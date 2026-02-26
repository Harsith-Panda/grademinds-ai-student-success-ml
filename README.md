# 🎓 GradeMinds AI: Intelligent Student Performance Prediction System

An AI-powered academic intelligence platform designed to predict student performance and pass/fail outcomes using Machine Learning.

GradeMinds AI leverages a robust ML pipeline and an interactive Streamlit web application to provide real-time academic performance predictions. Built using Scikit-learn and trained on structured student performance data, the system supports both regression (score prediction) and classification (pass/fail risk assessment).

---

## ✨ Key Features

### 📊 Dual Prediction System

- **Score Prediction (Regression):** Predicts Final Exam Score using Linear Regression.
- **Pass/Fail Classification:** Predicts whether a student will pass using Logistic Regression and Decision Tree models.

### 🧠 Multi-Model Architecture

- Linear Regression for continuous score prediction.
- Logistic Regression as classification baseline.
- Decision Tree Classifier with hyperparameter tuning for non-linear modeling.

### 📈 Performance Evaluation Dashboard

- R² Score, MAE, RMSE (Regression)
- Accuracy, Precision, Recall, F1-score (Classification)
- Confusion Matrix & Classification Reports
- Feature Importance (Decision Tree)

### ⚡ Real-Time Prediction Interface

- Interactive Streamlit form
- Instant prediction results
- Probability-based pass risk evaluation
- Clean, modern UI design

### 🔍 Data-Driven Insights

- Identifies key performance drivers
- Explains impact of attendance, GPA, and study hours
- Helps educators understand performance factors

---

## 🛠️ Tech Stack

**Frontend Application:** Streamlit  
**Machine Learning:** Scikit-learn

- Linear Regression
- Logistic Regression
- Decision Tree Classifier

**Data Processing:** Pandas, NumPy  
**Visualization:** Matplotlib, Seaborn  
**Serialization:** Pickle

---

## 📁 Project Structure

```
GradeMinds_AI/
├── app.py                               # Main Streamlit application
├── model.pkl                            # Serialized ML pipeline
├── requirements.txt                     # Dependencies
├── README.md                            # Project documentation
├── data/
│   ├── raw/
│   │   └── student_performance_raw.csv
│   └── processed/
│       └── student_performance_processed.csv
└── notebook/
    ├── eda_preprocessing.ipynb          # EDA & Data Cleaning
    ├── regression_model.ipynb           # Linear Regression
    └── classification_model.ipynb       # Logistic & Decision Tree
```

---

## 🔬 How It Works

### 1️⃣ Data Cleaning & Preprocessing (eda_preprocessing.ipynb)

- Removed non-relevant identifier columns
- Handled missing values using robust statistical methods
- Encoded categorical variables using One-Hot Encoding
- Created classification target (Pass_Fail)
- Produced a clean, reusable processed dataset

---

### 2️⃣ Model Training

#### 📘 Regression Model

- Linear Regression used to predict Final Exam Score
- Achieved:
  - **R² Score:** ~0.66
  - **MAE:** ~3.06
  - **RMSE:** ~3.77

This indicates strong predictive capability with low average error (~3–4 marks).

#### 📙 Classification Models

- Logistic Regression (Baseline)
- Decision Tree (Tuned)

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Models were compared and selected based on generalization performance.

---

### 3️⃣ Application Layer (app.py)

The Streamlit app:

- Accepts student inputs:
  - Study Hours Per Week
  - Attendance Rate
  - Previous Semester GPA
  - Extracurricular Involvement
- Applies preprocessing pipeline
- Generates:
  - Predicted Final Score
  - Pass/Fail Status
  - Probability of Passing
- Displays results in an interactive dashboard

---

## 📝 Input Features

The model uses the following features:

- `Study_Hours_Per_Week`
- `Attendance_Rate`
- `Previous_Semester_GPA`
- `Extracurricular_Involvement`
- Engineered features (if enabled):
  - Study-Attendance Interaction
  - Academic Consistency Score

Target Variables:

- `Final_Exam_Score` (Regression)
- `Pass_Fail` (Classification)

---

## 🚀 Setup & Installation

### 1️⃣ Clone the Repository

```
git clone <repository-url>
cd GradeMinds_AI
```

### 2️⃣ Create Virtual Environment (Recommended)

```
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Run the Application

Ensure `model.pkl` is in the root directory.

```
streamlit run app.py
```

The app will launch at:

```
http://localhost:8501
```

---

## 📊 Evaluation & Performance

### Regression Model

- R²: ~0.66
- MAE: ~3.06
- RMSE: ~3.77

### Classification Model

- High accuracy with stable generalization
- Balanced Precision & Recall
- Confusion Matrix available inside app

The system demonstrates strong predictive capability for structured academic datasets.

---

## 🎯 Project Highlights

✅ Clean ML pipeline  
✅ Baseline-first modeling strategy  
✅ Data leakage prevention  
✅ Model comparison methodology  
✅ Professional documentation  
✅ Deployment-ready application

---

## 🎓 Academic Value

GradeMinds AI demonstrates:

- End-to-end ML workflow
- Feature engineering exploration
- Regression & Classification modeling
- Model evaluation and comparison
- Deployment using Streamlit

---

# 🚀 GradeMinds AI – Turning Academic Data into Intelligent Insights
