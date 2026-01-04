#!/usr/bin/env python3
"""
Extract tuned model metrics using cross-validation predictions (not training data!)
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Paths
results_dir = '/Users/reembaraka/Documents/Boulder data science/DTSA 5506/Project/results'
data_dir = '/Users/reembaraka/Documents/Boulder data science/DTSA 5506/Project/Data/Pre-Processed'

# Load data
print("Loading data...")
X = pd.read_csv(f'{data_dir}/X.csv')
y = pd.read_csv(f'{data_dir}/y.csv')['Class']

# Set up cross-validation (same as notebooks)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Load tuned models
print("Loading tuned models...")
models = {}
with open(f'{results_dir}/best_lr_model.pkl', 'rb') as f:
    models['Logistic Regression'] = pickle.load(f)
with open(f'{results_dir}/best_svm_model.pkl', 'rb') as f:
    models['SVM'] = pickle.load(f)
with open(f'{results_dir}/best_xgb_model.pkl', 'rb') as f:
    models['XGBoost'] = pickle.load(f)

# Get cross-validation predictions (out-of-sample!)
print("Calculating metrics using cross-validation predictions...")
tuned_results = []

for model_name, model in models.items():
    # Get out-of-sample predictions using cross_val_predict
    y_pred = cross_val_predict(model, X, y, cv=skf, n_jobs=-1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = f1_score(y, y_pred)
    
    tuned_results.append({
        'Model': model_name,
        'Accuracy': f"{accuracy:.4f}",
        'Sensitivity': f"{sensitivity:.4f}",
        'Specificity': f"{specificity:.4f}",
        'Precision': f"{precision:.4f}",
        'F1-Score': f"{f1:.4f}",
        'Total_Errors': int(fp + fn),
        'False_Positives': int(fp),
        'False_Negatives': int(fn)
    })

# Create DataFrame and save
tuned_metrics_df = pd.DataFrame(tuned_results)
tuned_metrics_df.to_csv(f'{results_dir}/tuned_model_metrics.csv', index=False)

print("\n" + "="*90)
print("TUNED MODEL PERFORMANCE METRICS (Cross-Validation)")
print("="*90)
print(tuned_metrics_df.to_string(index=False))
print("\n✅ Saved to: tuned_model_metrics.csv")
print("="*90)
