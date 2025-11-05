# 💳 Credit Scoring Model

### 🧠 Objective
The objective of this project is to **predict an individual's creditworthiness** (Good or Bad Credit) using their **past financial data**.  
By analyzing key financial indicators, we can classify whether a person is likely to **repay loans responsibly** or **default**.

---

## 🏗️ Approach
We used **Machine Learning classification algorithms** to predict credit scores based on features such as income, debts, and payment history.

### Algorithms Used:
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  

### Model Evaluation Metrics:
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  

---

## 📊 Dataset Description
The dataset contains financial details such as:

| Feature | Description |
|----------|-------------|
| **Income** | Annual income of the individual |
| **Age** | Age of the individual |
| **LoanAmount** | Total loan amount taken |
| **CreditHistory** | Whether the individual has a clean credit history (1 = Yes, 0 = No) |
| **DebtRatio** | Ratio of monthly debt payments to income |
| **LatePayments** | Number of times payment was delayed |
| **CreditScore** | Target Variable (1 = Good, 0 = Bad) |

📁 The dataset used here is **synthetic** (randomly generated for demonstration), but the model works with any real-world financial dataset.

---

## ⚙️ Steps Followed

### 1️⃣ Data Preparation
- Imported required libraries (`pandas`, `numpy`, `sklearn`)
- Created or loaded dataset
- Split into **training (80%)** and **testing (20%)** data

### 2️⃣ Data Preprocessing
- Normalized feature values using `StandardScaler`
- Removed target variable from feature set

### 3️⃣ Model Training
Trained three ML models:
- Logistic Regression  
- Decision Tree  
- Random Forest  

### 4️⃣ Model Evaluation
Used metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC to assess performance.

### 5️⃣ Model Comparison
Compared all models and selected the **best-performing** one (based on F1-Score).

### 📈 Sample Output
```
Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	0.74	0.72	0.75	0.73	0.74
Decision Tree	0.78	0.77	0.79	0.78	0.79
Random Forest	0.84	0.83	0.85	0.84	0.86
```

### 🏆 Best Model: Random Forest (Highest F1-Score)

### 📘 Conclusion

The Credit Scoring Model effectively predicts whether an individual is creditworthy.

Among all algorithms tested, the Random Forest Classifier performed best with the highest F1-Score and ROC-AUC values.

This model can assist banks and financial institutions in automated credit evaluation, reducing manual risk assessment.

### 🧑‍💻 Technologies Used

Python

Scikit-learn

Pandas & NumPy

Matplotlib (optional for visualization)