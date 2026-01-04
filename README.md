# Chronic Kidney Disease Early Detection: A Comparative Analysis of Classification Models

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Chronic kidney disease (CKD) affects 850+ million people globally, with 90% unaware they have it. This project trained and compared five classification models to predict CKD in early stages using routine blood and urine tests from 400 patient records. All tuned models achieved clinical targets (Sensitivity >95%, Specificity >90%), with Logistic Regression recommended for deployment due to optimal balance of performance and interpretability.

**Impact:** Enable early CKD detection in primary care settings, preventing progression to dialysis ($90,000+/year per patient).

---

## Results Summary

| Model | Errors | Sensitivity | Specificity | F1-Score |
|-------|--------|-------------|-------------|----------|
| **Logistic Regression (Recommended)** | 7 | 98.67% | 98.00% | 0.9774 |
| SVM | 5 | 98.67% | 98.80% | 0.9867 |
| XGBoost | 8 | 97.33% | 99.20% | 0.9851 |

**Statistical Validation:** McNemar's test confirmed no significant difference between top models (p > 0.05), making interpretability the deciding factor.

**Key Finding:** SHAP analysis identified hemoglobin, specific gravity, packed cell volume, and albumin as most predictive features, aligning with clinical knowledge of CKD pathophysiology.

---

## Technical Approach

**Data:** 400 patients, 24 clinical features, 10.12% missing values  
**Preprocessing:** kNN imputation (k=5), StandardScaler, retained outliers (clinical significance)  
**Modeling:** 5 algorithms with 5-fold stratified cross-validation, scikit-learn pipelines  
**Optimization:** GridSearchCV with F1-scoring (balanced sensitivity/specificity)  
**Validation:** McNemar's test, SHAP analysis, validation curves

**Critical Design Decision:** Switched from recall-only optimization (37 errors, 85% specificity) to F1-score optimization (≤8 errors, ≥98% specificity), demonstrating understanding of clinical trade-offs.

---

## Key Visualizations

### Model Performance
![ROC Curves](Results/ROC_Curves.png)
*ROC curves comparison with zoomed inset - all models achieve >0.95 AUC*

![Confusion Matrices](Results/Confusion_Matrices_All_Models.png)
*Confusion matrices showing error breakdown for all five models*

### Feature Importance (SHAP Analysis)
![Feature Comparison](Results/Feature_Importance_Comparison.png)
*Top 15 features across four models - consistent importance of anemia markers*

![SHAP Beeswarm - Logistic Regression](Results/shap_beeswarm_lr.png)
*SHAP beeswarm plot showing feature value impact on predictions*

### Data Quality & Preprocessing
![Correlation Matrix](Results/Correlation_Matrix_After_kNN_Imputation.png)
*Feature correlations after kNN imputation - Hemoglobin (0.77), PCV (0.74), RBC (0.70)*

*Full analysis with 18 visualizations and 5 CSV files available in `/Results/` folder.*

---

## Skills Demonstrated

**Machine Learning:** Classification models (LR, SVM, XGBoost, Random Forest, Decision Tree), hyperparameter tuning, cross-validation, pipeline design

**Statistical Analysis:** McNemar's test, correlation analysis, feature importance, model comparison, handling class imbalance

**Data Processing:** Missing value imputation (kNN), feature encoding, scaling, outlier analysis

**Model Interpretability:** SHAP analysis across multiple models, clinical validation of features

**Domain Knowledge:** Healthcare analytics, understanding clinical trade-offs (false negatives vs false positives), translating ML results to medical context

**Tools:** Python, scikit-learn, SHAP, pandas, numpy, matplotlib, seaborn, XGBoost, Jupyter Notebooks

---

## Project Structure

```
├── Notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb
│   ├── 02_Model_Training_and_Evaluation.ipynb
│   └── 03_SHAP_Analysis_and_Interpretation.ipynb
├── Data/
│   └── ckd.csv (UCI ML Repository)
└── Results/
    ├── 18 visualization files
    └── 5 CSV analysis files
```

---

## Key Insights

**Model-Dependent Feature Importance:** Serum Creatinine ranked #1 in tree models but #11 in Logistic Regression due to non-linear relationships with outliers—tree models handle this naturally while linear models don't. This demonstrates the value of comparing multiple model types.

**Balanced Optimization Matters:** F1-score optimization achieved 98.67% sensitivity and 98.00% specificity. Initial recall-only optimization produced 100% sensitivity but dropped specificity to 85%, resulting in 37 errors (clinically unacceptable).

**Interpretability vs Performance:** Despite SVM catching 2 more errors, Logistic Regression was recommended for deployment due to clear coefficient interpretation, crucial for physician trust and clinical adoption.

---

## Dataset

**Source:** UCI Machine Learning Repository - Apollo Hospitals, India  
**Link:** https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease  
**Size:** 400 patients, 24 clinical features (blood and urine test markers)  
**Target:** CKD (62.5%) vs Not CKD (37.5%)

---

## Contact

**Reem Baraka**  
Data Science Graduate Student | University of Colorado Boulder  
[GitHub](https://github.com/RBaraka) | [LinkedIn](https://www.linkedin.com/in/reem-baraka)

---

*Complete technical report and presentation available in repository.*
