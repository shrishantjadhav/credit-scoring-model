import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

np.random.seed(42)
data = {
    'Income': np.random.randint(20000, 100000, 500),
    'Age': np.random.randint(18, 65, 500),
    'LoanAmount': np.random.randint(1000, 50000, 500),
    'CreditHistory': np.random.choice([0, 1], 500),
    'DebtRatio': np.random.uniform(0.1, 1.0, 500),
    'LatePayments': np.random.randint(0, 10, 500),
    'CreditScore': np.random.choice([0, 1], 500)  # Target variable
}

df = pd.DataFrame(data)
print("✅ Dataset Loaded Successfully\n")
print(df.head())

X = df.drop('CreditScore', axis=1)
y = df['CreditScore']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    results[name] = [accuracy, precision, recall, f1, roc_auc]
    print(f"\n🔹 {name} Performance:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

results_df = pd.DataFrame(results, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
print("\n📈 Model Comparison:\n")
print(results_df)

best_model_name = results_df.loc['F1-Score'].idxmax()
print(f"\n🏆 Best Model Based on F1-Score: {best_model_name}")
