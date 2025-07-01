import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , ConfusionMatrixDisplay,confusion_matrix, recall_score, precision_recall_curve,precision_score,classification_report
import matplotlib.pyplot as plt
#importing and processing dataset
heart_data = pd.read_csv('heart.csv')
heart_data = heart_data.drop_duplicates()
x=heart_data.drop(columns='target',axis=1)
y=heart_data['target']
#spliting into train and test data
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)
#model trainning logistic regression
model = LogisticRegression()
model.fit(x_train,y_train)
#Predict Probabilities
y_proba = model.predict_proba(x_test)[:, 1]
#Precision Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
#Choose Best Threshold with 100% Recall and Max Precision
best_threshold = None
best_precision = 0

for p, r, t in zip(precision, recall, thresholds):
    if r >= 1 and p > best_precision:
        best_precision = p
        best_threshold = t

print(f"Best threshold with recall=1.0: {best_threshold:.4f}, precision: {best_precision:.4f}")
#Predict using custom Threshold
def model_predict_with_threshold(x,model = model , threshold = best_threshold ) :
  y_proba = model.predict_proba(x)[:,1]
  return (y_proba>=threshold).astype(int)
x_test_prediction=model_predict_with_threshold(x_test)
#Model Evaluation
cm = confusion_matrix(y_test, x_test_prediction)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, x_test_prediction))
# Confusion Matrix Plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(6, 6))  # Create custom Axes
disp.plot(cmap="Reds", ax=ax)          # Plot on custom Axes
ax.set_title("Confusion Matrix")       # ✅ Now you can set title here
plt.savefig("visuals/confusion_matrix.png")
plt.close()
# Precision-Recall vs Threshold Plot
plt.figure(figsize=(8, 5))
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.axvline(best_threshold, color='gray', linestyle='--', label=f"Chosen Threshold ({best_threshold:.2f})")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision & Recall vs Threshold")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("visuals/precision_recall_curve.png")
plt.close()