# Results

All outputs from the CKD prediction analysis, organized by type.

## Directory Structure

```
results/
├── data/                          # CSV files with analysis results
├── visualizations/
│   ├── exploratory/              # EDA plots (Notebook 1)
│   ├── performance/              # Model evaluation plots (Notebook 2)
│   ├── feature_importance/       # SHAP analysis plots (Notebook 3)
│   └── optimization/             # Hyperparameter tuning plots (Notebook 3)
└── models/                        # Trained models (excluded from git)
```

## Data Files (`data/`)

**Notebook 1 - EDA:**
- `dataset_summary_table.csv` - Dataset statistics
- `missing_data_summary.csv` - Missing data patterns

**Notebook 2 - Training:**
- `model_comp_results.csv` - Model comparison (default parameters)
- `detailed_model_metrics.csv` - Formatted metrics with thresholds

**Notebook 3 - Optimization:**
- `hyperparameter_tuning_summary.csv` - Best hyperparameters
- `tuned_model_metrics.csv` - Final model performance (optimized)

## Visualizations

### Exploratory Analysis (`visualizations/exploratory/`)
- `class_balance.png` - Target variable distribution
- `Missing_Data_HeatMap.png` - Missing data patterns
- `Correlation_matrix_of_numeric_features.png` - Pre-imputation correlations
- `Distribution_of_Key_Numeric_Features_by_CKD_Status.png` - Feature distributions
- `Categorical_Features_Distribution_by_CKD_Status.png` - Categorical patterns
- `Serum_Creatinine_Distribution_Analysis.png` - Key biomarker analysis
- `Correlation_Matrix_After_kNN_Imputation.png` - Post-imputation correlations

### Model Performance (`visualizations/performance/`)
- `ROC_Curves.png` - ROC curves comparison
- `Confusion_Matrices_All_Models.png` - All model confusion matrices

### Feature Importance (`visualizations/feature_importance/`)
- `Feature_Importance_Logistic_Regression_SHAP.png` - SHAP bar plot
- `Feature_Importance_Comparison.png` - Cross-model comparison (top 15)
- `shap_beeswarm_lr.png` - Logistic Regression beeswarm
- `shap_beeswarm_rf.png` - Random Forest beeswarm
- `shap_beeswarm_xgb.png` - XGBoost beeswarm
- `shap_beeswarm_svm.png` - SVM beeswarm

### Optimization & Testing (`visualizations/optimization/`)
- `hyperparameter_validation_curves.png` - Validation curves (6 parameters)
- `xgboost_estimators_comparison.png` - n_estimators optimization
- `mcnemar_test_visualization.png` - Statistical model comparison

## Model Files (`models/`)

*Excluded from version control:*
- `best_lr_model.pkl` - Logistic Regression (2.2 KB)
- `best_svm_model.pkl` - SVM (11 KB)  
- `best_xgb_model.pkl` - XGBoost (193 KB)

---

*Generated from notebooks 1-3. Fixed random seeds ensure consistent results.*
