# Dataset Information

## Source
**UCI Machine Learning Repository**  
https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease

**Collection:** Apollo Hospitals, India (July - November 2015)

---

## Dataset Statistics

- **Total Patients:** 400
- **CKD Cases:** 250 (62.5%)
- **Not CKD Cases:** 150 (37.5%)
- **Features:** 24 clinical features + 1 target variable
- **Missing Data:** 10.12% overall

---

## Feature Dictionary

### Demographic
| Feature | Type | Description | Units |
|---------|------|-------------|-------|
| Age | Numeric | Patient age | years |

### Blood Pressure
| Feature | Type | Description | Units |
|---------|------|-------------|-------|
| Blood Pressure | Numeric | Blood pressure measurement | mm/Hg |

### Kidney Function Tests
| Feature | Type | Description | Units |
|---------|------|-------------|-------|
| Specific Gravity | Categorical | Urine concentration (1.005, 1.010, 1.015, 1.020, 1.025) | - |
| Albumin | Categorical | Protein in urine (0, 1, 2, 3, 4, 5) | - |
| Sugar | Categorical | Glucose in urine (0, 1, 2, 3, 4, 5) | - |
| Blood Urea | Numeric | Urea nitrogen in blood | mg/dL |
| Serum Creatinine | Numeric | Creatinine in blood | mg/dL |
| Sodium | Numeric | Sodium level | mEq/L |
| Potassium | Numeric | Potassium level | mEq/L |

### Blood Tests
| Feature | Type | Description | Units |
|---------|------|-------------|-------|
| Hemoglobin | Numeric | Oxygen-carrying protein | gms |
| Packed Cell Volume | Numeric | Red blood cell volume | % |
| White Blood Cell Count | Numeric | WBC count | cells/cumm |
| Red Blood Cell Count | Numeric | RBC count | millions/cmm |

### Clinical Signs
| Feature | Type | Description | Values |
|---------|------|-------------|--------|
| Red Blood Cells | Categorical | RBCs in urine | normal, abnormal |
| Pus Cell | Categorical | Pus cells in urine | normal, abnormal |
| Pus Cell Clumps | Categorical | Pus cell clusters | present, not present |
| Bacteria | Categorical | Bacteria in urine | present, not present |

### Medical History
| Feature | Type | Description | Values |
|---------|------|-------------|--------|
| Hypertension | Categorical | High blood pressure history | yes, no |
| Diabetes Mellitus | Categorical | Diabetes history | yes, no |
| Coronary Artery Disease | Categorical | Heart disease history | yes, no |
| Appetite | Categorical | Appetite status | good, poor |
| Pedal Edema | Categorical | Ankle swelling | yes, no |
| Anemia | Categorical | Low hemoglobin | yes, no |

### Target Variable
| Feature | Type | Description | Values |
|---------|------|-------------|--------|
| Class | Categorical | CKD diagnosis | ckd (0), notckd (1) |

---

## Data Quality

### Missing Data by Feature (Top 10)
1. Red Blood Cells: 38.8%
2. Pus Cell: 16.2%
3. Pus Cell Clumps: 10.0%
4. Bacteria: 10.0%
5. Blood Glucose Random: 11.0%
6. Blood Urea: 4.8%
7. Serum Creatinine: 4.5%
8. Sodium: 22.0%
9. Potassium: 22.2%
10. Hemoglobin: 13.0%

**Handling:** kNN imputation (k=5) used to preserve feature relationships

---

## Data Processing

### Encoding
- **Target Variable:** ckd → 0, notckd → 1
- **Binary Categorical:** yes/no → 1/0
- **Ordinal Categorical:** LabelEncoder for ordered categories
- **Nominal Categorical:** One-hot encoding (if applicable)

### Scaling
- **StandardScaler** applied to all numeric features
- Performed within cross-validation pipeline to prevent data leakage

### Outliers
- **Detection:** IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
- **Action:** Retained (clinical data - extreme values are medically significant)

---

## Clinical Significance

### Top Predictive Features
1. **Hemoglobin** - Anemia is common in CKD (reduced erythropoietin)
2. **Specific Gravity** - Reflects kidney's ability to concentrate urine
3. **Packed Cell Volume** - Another anemia indicator
4. **Albumin** - Proteinuria indicates kidney damage
5. **Serum Creatinine** - Waste product, rises with kidney failure

---

## Usage

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/ckd_full.csv')

# Basic info
print(df.shape)  # (400, 25)
print(df['Class'].value_counts())
```

---

## Citation

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

**Original Data Contributors:**  
Dr. P. Soundarapandian, M.D., D.M (Senior Consultant Nephrologist)  
Apollo Hospitals, Managiri, Madurai Main Road, Karaikudi, Tamil Nadu, India
