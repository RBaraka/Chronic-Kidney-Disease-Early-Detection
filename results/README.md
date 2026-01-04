# Results

All outputs from the CKD prediction analysis.

## Data Files

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

**Exploratory Analysis:**
- `class_balance.png` - Target variable distribution
- `Missing_Data_HeatMap.png` - Missing data patterns visualization
- `Correlation_matrix_of_numeric_features.png` - Feature correlations (pre-imputation)
- `Distribution_of_Key_Numeric_Features_by_CKD_Status.png` - Numeric feature distributions
- `Categorical_Features_Distribution_by_CKD_Status.png` - Categorical feature patterns
- `Serum_Creatinine_Distribution_Analysis.png` - Detailed analysis of key biomarker
- `Correlation_Matrix_After_kNN_Imputation.png` - Post-imputation correlations

**Model Performance:**
- `ROC_Curves.png` - ROC curves comparison
- `Confusion_Matrices_All_Models.png` - All model confusion matrices

**Feature Importance:**
- `Feature_Importance_Logistic_Regression_SHAP.png` - SHAP bar plot (Logistic Regression)
- `Feature_Importance_Comparison.png` - Cross-model comparison (top 15 features)
- `shap_beeswarm_lr.png` - SHAP beeswarm (Logistic Regression)
- `shap_beeswarm_rf.png` - SHAP beeswarm (Random Forest)
- `shap_beeswarm_xgb.png` - SHAP beeswarm (XGBoost)
- `shap_beeswarm_svm.png` - SHAP beeswarm (SVM)

**Optimization & Testing:**
- `hyperparameter_validation_curves.png` - Validation curves (6 hyperparameters)
- `xgboost_estimators_comparison.png` - n_estimators optimization
- `mcnemar_test_visualization.png` - Statistical model comparison

## Model Files

*Excluded from version control (see `.gitignore`):*
- `best_lr_model.pkl` - Logistic Regression (2.2 KB)
- `best_svm_model.pkl` - SVM (11 KB)  
- `best_xgb_model.pkl` - XGBoost (193 KB)

---

*Generated from notebooks 1-3. Fixed random seeds ensure consistent results.*
