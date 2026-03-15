# 🔧 Predictive Modelling of Truck Air Pressure System (APS) Failure

**MSc Business Analytics Dissertation — University of Kent, 2024**  
**Author:** Sidharth Dasan  
**Grade:** Merit (2:1)  

📧 sidharth.anila.dasan@gmail.com · 🔗 [LinkedIn](https://www.linkedin.com/in/sidharth-dasan) · 💻 [GitHub](https://github.com/sidharth-dasan)

---

## 📌 Project Overview

This project develops an end-to-end machine learning pipeline to **predict Air Pressure System (APS) failures in Scania trucks** before they occur — enabling proactive maintenance and reducing costly unplanned downtime.

### Business Problem
Truck fleets in the US move approximately **70% of all freight tonnage**. Unplanned APS failures result in:
- Emergency repair costs significantly higher than scheduled maintenance
- Unplanned downtime reducing fleet availability by up to 20%
- Cascading supply chain disruptions

By predicting failures in advance, fleet operators can shift from **reactive to proactive maintenance** — optimising resource allocation and reducing costs.

---

## 📊 Dataset

**Source:** UCI Machine Learning Repository — APS Failure at Scania Trucks  
🔗 https://doi.org/10.24432/C51S51

| Property | Detail |
|---|---|
| Observations | 60,000 records |
| Features | 171 anonymised sensor readings |
| Target | Binary — `neg` (no failure) / `pos` (APS failure) |
| Class ratio | ~98.3% negative / ~1.7% positive |
| Missing values | ~37% across features |

> **Note:** The notebook uses a synthetic dataset that replicates the statistical properties of the original UCI data (class imbalance ratio, missing value patterns, feature distributions). To run on the real data, download from the UCI link above and replace the synthetic generation block in Section 2.

---

## 🛠️ Technical Pipeline

```
Raw Sensor Data (60,000 records, 171 features)
        ↓
1. Feature Engineering   — Binning aggregation + label encoding
        ↓
2. Missing Value Handling — Median imputation (robust to outliers)
        ↓
3. Correlation Analysis  — Remove multicollinear features (|r| > 0.95)
        ↓
4. Feature Scaling       — StandardScaler (mean=0, std=1)
        ↓
5. PCA                   — Dimensionality reduction (95% variance retained)
        ↓
6. SMOTEENN              — Class imbalance handling
        ↓
7. Model Training        — Logistic Regression / Decision Tree / Random Forest
        ↓
8. Evaluation            — Accuracy, F1, Kappa, Economic Cost Framework
```

---

## 🤖 Models Evaluated

| Model | Purpose |
|---|---|
| **Logistic Regression** | Interpretable linear baseline |
| **Decision Tree** | Explainable rule-based classifier |
| **Random Forest** | Ensemble method — robust to noise and imbalance |

---

## 📈 Key Results

| Model | Accuracy | Cohen's Kappa | Total Cost |
|---|---|---|---|
| Logistic Regression | ~91% | ~0.79 | Highest |
| Decision Tree | ~94% | ~0.87 | Medium |
| **Random Forest** ✅ | **97.24%** | **0.9439** | **Lowest (10,900)** |

### Economic Cost Framework
A key contribution of this project is evaluating models using a **real-world cost structure** rather than accuracy alone:

| Error Type | Real-World Consequence | Cost Weight |
|---|---|---|
| **False Negative** (missed failure) | Undetected APS failure → breakdown, downtime | **500 units** |
| **False Positive** (false alarm) | Unnecessary maintenance check | **10 units** |

**Random Forest achieved the lowest total misclassification cost of 10,900 units** — making it the most economically optimal model for deployment.

---

## 🔬 Key Techniques

### SMOTEENN — Class Imbalance Handling
The dataset suffers from severe class imbalance (~98% normal, ~2% failure). SMOTEENN addresses this by:
- **SMOTE** — Generates synthetic minority class (failure) samples via interpolation
- **ENN (Edited Nearest Neighbors)** — Removes noisy and borderline samples from both classes

Without SMOTEENN, models default to predicting "no failure" for almost all inputs — achieving high accuracy but missing nearly every actual failure.

### PCA — Dimensionality Reduction
With 171 sensor features, high dimensionality increases computational cost and risks overfitting. PCA:
- Reduced dimensionality by ~40%
- Retained 95% of total variance
- Improved model generalisation and training speed

### Median Imputation
Real-world sensor data contains significant missing values (~37%). Median imputation was chosen over mean imputation as it is **robust to the skewed distributions** typical of sensor readings.

---

## 📁 Repository Structure

```
aps-failure-prediction/
│
├── APS_Failure_Prediction.ipynb    # Full pipeline notebook
├── README.md                       # This file
```

---

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)
Click the badge below to open directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sidharth-dasan/aps-failure-prediction/blob/main/APS_Failure_Prediction.ipynb)

### Option 2 — Local Setup
```bash
# Clone the repository
git clone https://github.com/sidharth-dasan/aps-failure-prediction.git
cd aps-failure-prediction

# Install dependencies
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn jupyter

# Launch notebook
jupyter notebook APS_Failure_Prediction.ipynb
```

### Dependencies
```
numpy
pandas
scikit-learn
imbalanced-learn    # For SMOTEENN
matplotlib
seaborn
jupyter
```

---

## 💡 Business Impact

This research demonstrates that ML-based predictive maintenance can:

✅ Reduce maintenance-related downtime by up to **20%**  
✅ Shift fleet management from a **cost centre to a strategic asset**  
✅ Enable **proactive resource allocation** and parts inventory planning  
✅ Provide **economically optimal** predictions through cost-weighted evaluation  

---

## 🔭 Future Work

- Real-time integration with IoT sensor streams for live prediction
- Deep learning models (LSTM/GRU) for time-series sensor data
- SHAP values for model explainability and maintenance team interpretation
- Integration of environmental features (weather, route, driver behaviour)
- Deployment as a REST API for fleet management system integration

---

## 📚 References

- UCI Machine Learning Repository — APS Failure at Scania Trucks (2017): https://doi.org/10.24432/C51S51
- Chawla et al. (2004) — SMOTE: Synthetic Minority Over-sampling Technique. SIGKDD Explorations
- Nishat et al. (2022) — SMOTE-ENN and Hyperparameter Optimisation. Scientific Programming
- Jolliffe (2005) — Principal Component Analysis. Encyclopedia of Statistics in Behavioral Science
- Yangalasetty Lokesh et al. (2020) — Truck APS Failure Detection using Machine Learning. IEEE

---

*MSc Business Analytics — University of Kent, 2024*  
*Sidharth Dasan · sidharth.anila.dasan@gmail.com*
