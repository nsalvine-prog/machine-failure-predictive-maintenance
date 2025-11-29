# Machine Failure – Predictive Maintenance  


## Business Problem

Perusahaan X is a motor–vehicle spare-parts manufacturer (bearings, crank shaft, piston, etc.) supplying several automotive brands in Indonesia.  

The company wants to **expand to international markets**, but currently faces:

- Frequent **machine failures during production**
- High **unexpected maintenance costs**
- Concerns from management that the plant is **not ready to increase capacity**

Machine failures are categorized into five **independent failure types**:

- **TWF** – Tool Wear Failure  
- **HDF** – Heat Dissipation Failure  
- **PWF** – Power Failure  
- **OSF** – Overstrain Failure  
- **RNF** – Random Failure  

To support expansion, the production director requests a **predictive maintenance model** that can anticipate machine failures before they occur.

---

## Goals

1. **Build a predictive maintenance model** that accurately classifies whether a machine will fail, based on sensor and process features.
2. **Provide maintenance recommendations** based on model predictions to:
   - Reduce unplanned downtime  
   - Lower maintenance costs  
   - Enable capacity expansion and international market entry  

---

## Data Understanding

Dataset: **Predictive Maintenance Dataset** (10,000 rows × 14 columns)

**Main features:**

- `UDI` – Unique Device Identifier (1–10,000)  
- `Product ID` – Product code (alphanumeric)  
- `Type` – Machine type: **L (Low), M (Medium), H (High)**  
- `Air temperature [K]`  
- `Process temperature [K]`  
- `Rotational speed [rpm]`  
- `Torque [Nm]`  
- `Tool wear [min]`  
- `Machine failure` – target (0 = no failure, 1 = failure)

**Failure indicator columns:**

- `TWF` – Tool wear failure  
- `HDF` – Heat dissipation failure  
- `PWF` – Power failure  
- `OSF` – Overstrain failure  
- `RNF` – Random failure  

**Initial data quality:**

- No duplicate rows  
- No missing values  
- Mixed data types; some columns required conversion  
- Categorical variables with high cardinality (`Product ID`)

---

## Data Cleaning

Key cleaning operations:

- Converted `Product ID` to object type and filled unknown values  
- Standardized column names to **lowercase_with_underscores**  
- Verified no missing values and no duplicates  
- Kept outliers because the task is **anomaly / failure detection**  
- Confirmed strong class imbalance:  
  - ~9,661 non-failure vs 339 failure records  

---

## Exploratory Data Analysis (EDA)

### Machine Type Distribution
- Type **L** ≈ 60%  
- Type **M** ≈ 30%  
- Type **H** ≈ 10%  

### Outliers & Imbalance
- Outliers in `rotational_speed_rpm` and `torque_nm`  
- Strongly imbalanced target: most machines do **not** fail  

### Failure Indicators
From bar plots of `TWF`, `HDF`, `PWF`, `OSF`, `RNF`:

- Failures are **rare events** in each failure type  
- HDF, PWF, and OSF contribute noticeable but small portions of total failures  

### Boxplots (by `machine_failure`)
Key observations:

- Different distributions of `rotational_speed_rpm`, `torque_nm`, `tool_wear_min`, and temperatures between failure / non-failure cases  
- Failures often occur at:
  - Lower rotational speed regions and certain torque ranges  
  - Higher tool wear and specific temperature differences  

### Automated EDA (dataprep)

- 14 variables, 10,000 rows  
- 6 numerical, 8 categorical  
- `air_temperature_k` and `process_temperature_k` strongly positively correlated (≈ 0.88)  
- `rotational_speed_rpm` and `torque_nm` strongly negatively correlated (≈ −0.88)  

---

## Feature Engineering

**Dropped low-value / redundant columns:**

- `product_id`, `udi`, `twf`, `osf`, `rnf`, `hdf`, `pwf`

**Outlier-based features:**

- Flags for high rotational speed and torque:  
  - `>95_rpm`, `>99_rpm`, `>95_torque`, `>99_torque`  

**Process-based engineered features:**

- `rotational_speed_rad` – RPM → rad/s  
- `power` – rotational speed × torque  
- `power_wear` – `power × tool_wear`  
- `overstrain` – `tool_wear × torque`  
- `temperature_difference` – `process_temperature_k − air_temperature_k`  
- `temperature_power` – `temperature_difference / power`  
- Interaction terms with type:  
  - `overstrain_typeL`, `overstrain_typeM`, `overstrain_typeH`  
- `rotational_speed_wear` – `rotational_speed_rpm × tool_wear_min`  

**Rule-based flags from EDA insights:**

- `temp_diff<8koma6k` – temperature difference < 8.6 K  
- `rpm<1380`, `rpm>2500` – extreme rotational speeds  
- `power<3500`, `power>9000` – extreme power regions  
- `overstrain>11000` – high overstrain  
- `torque>65nm` – high torque  

**Encoding:**

- One-hot encoding for `type` → `type_L`, `type_M`, `type_H`

---

## Modeling Approach

### Tools & Libraries

- Python, Pandas, NumPy, Matplotlib, Seaborn, Plotly  
- scikit-learn (KMeans, train_test_split, MinMaxScaler, metrics)  
- imbalanced-learn (SMOTE – tested but not used in final model)  
- LightGBM (for feature importance)  
- **PyCaret – classification** (automated ML & model comparison)  
- SHAP (model explainability, optional)  

### Handling Imbalance

- Tried several resampling strategies (e.g., SMOTE)  
- Performance gain was **minimal (< 0.2%)**  
- Final model uses **original imbalanced data** with stratified cross-validation

### PyCaret Setup

- Target: `machine_failure`  
- `train_size = 0.7`, `test_size = 0.3`  
- `fold_strategy = 'stratifiedkfold'`, `fold = 10`  
- **Feature selection** with:
  - `feature_selection = True`  
  - `n_features_to_select = 13`  
  - `feature_selection_method = 'classic'` (SelectFromModel)  
  - `feature_selection_estimator = 'lightgbm'`  
- Added custom metric: **False Positive Rate (FPR)**  

### Model Comparison

Multiple models were compared (e.g., Random Forest, LightGBM, CatBoost, Logistic Regression).  
Based on metrics (AUC, F1, recall, precision), the best base model was:

> **Gradient Boosting Classifier (GBC)**

**Why GBC?**

- High **recall**, **precision**, **F1**, and **AUC**  
- Handles complex, non-linear relationships  
- Works well with imbalanced data and tabular features  

---

## Hyperparameter Tuning & Ensembles

### Automatic Tuning

- PyCaret tuning focused on optimizing **F1**, **precision**, and **recall**  
- Result: **base GBC model already strong**; tuning did not significantly improve performance

### Custom Grid Search

Two-stage grid search over:

- `n_estimators`, `max_depth`, `subsample`  
- `min_samples_split`, `min_samples_leaf`  

Produced a tuned GBC with stable training vs validation scores (good fit).

### Ensembles Tested

- **Bagging** on tuned GBC  
- **Boosting** on tuned GBC  
- **Blending** with:
  - LR + DT + KNN  
  - Tuned GBC + Random Forest  

Result:  
The best trade-off between performance and computation was **tuned GBC (custom grid search)**.

---

## Evaluation

### Key Metrics (Test Set)

- **AUC-ROC ≈ 0.99** for both classes (0 & 1)  
- Excellent ability to distinguish failure vs non-failure  

### Confusion Matrix (example interpretation)

- **TP = 84** – correctly predicted failures  
- **TN = 2896** – correctly predicted non-failures  
- **FP = 2** – predicted failure, actually not failed  
- **FN = 18** – predicted non-failure, actually failed  

### Classification Report (Example)

**Class 0 – No Failure**

- Precision ≈ 0.993  
- Recall ≈ 0.999  
- F1-Score ≈ 0.996  

**Class 1 – Failure**

- Precision ≈ 0.976  
- Recall ≈ 0.814  
- F1-Score ≈ 0.888  

This indicates:

- Very low **false alarm rate** (few FPs)  
- Good **detection rate** for actual failures (recall ≈ 81%)  

### Learning Curve

- Model shows **low bias** and **acceptable variance**  
- Training and validation scores converge with more data, indicating good generalization  

---

## Conclusions & Recommendations

### Conclusions

- A **Gradient Boosting Classifier with custom tuning** delivers strong performance for predictive maintenance.  
- The model can accurately identify most failing machines while keeping false alarms extremely low.  
- Engineered features such as **power**, **overstrain**, **temperature difference**, and extreme-value flags greatly improve predictive power.

### Recommendations

1. **Deploy the model** in production to monitor machine conditions in real time.  
2. Use model predictions to:
   - Schedule **preventive maintenance**  
   - Inspect machines flagged with high failure probability  
   - Prioritize resources for high-risk production lines  
3. Log predictions and actual outcomes to **continuously retrain and improve** the model.  
4. Investigate business trade-offs between:
   - Reducing **false negatives** (missed failures)  
   - Keeping **false positives** (unnecessary maintenance) at an acceptable cost  

---

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
