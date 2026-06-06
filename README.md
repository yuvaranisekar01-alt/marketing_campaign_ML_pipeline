# 📊 Marketing Campaign ML Pipeline
### Nykaa · Purplle · Tira — Campaign Intelligence System

---

## 🎯 Project Overview

An end-to-end Machine Learning pipeline built on marketing campaign data 
from three major Indian beauty brands — Nykaa, Purplle, and Tira.

The pipeline predicts:
- 💰 **Revenue** — How much revenue will a campaign generate? (Regression)
- 📈 **Profit/Loss** — Will this campaign be profitable? (Classification)

---

## 🏗️ Project Structure
---

## 📊 Dataset

| Brand | Rows | Source |
|---|---|---|
| Nykaa | ~50,000 | CSV |
| Purplle | ~50,000 | CSV |
| Tira | ~50,000 | CSV |
| **Combined** | **~150,000** | merged |

**Features:** Campaign Type, Channel Used, Target Audience, 
Impressions, Clicks, Leads, Conversions, Acquisition Cost, 
Engagement Score, ROI, Revenue, Date

---

## 🔬 ML Pipeline Steps

### 1. Data Ingestion & Preprocessing
- Merged 3 brand datasets
- Handled missing values — median for numeric, proportionate sampling for categorical
- Fixed Channel_Used multi-value column
- Saved to PostgreSQL database

### 2. Feature Engineering
- Extracted Month, Year, Quarter from Date
- Created ratio features — CTR, Conversion Rate, Cost Per Click
- Created target column `Profit_Loss` from ROI

### 3. Feature Selection
Ran 5 tests to select best features:

| Test | Purpose |
|---|---|
| Pearson Correlation | numeric vs numeric relationship |
| Chi-Square | categorical vs target relationship |
| T-test | numeric vs categorical target |
| VIF | multicollinearity check |
| Feature Importance | Random Forest importance scores |

**Final selected features (6):**

### 4. Model Training

**Regression — Revenue Prediction:**

| Model | Test R² |
|---|---|
| Linear Regression | 0.23 |
| Random Forest | 0.70 |
| **XGBoost ✅** | **0.70** |

**Classification — Profit/Loss Prediction:**

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 88% | 0.83 |
| Random Forest | 90% | 0.85 |
| **XGBoost ✅** | **90%** | **0.85** |

### 5. Model Evaluation

**Regression Results:**
Test R²  : 0.70
MAE      : ₹1,68,540
RMSE     : ₹2,58,839

**Classification Results:**
Accuracy : 90%
Loss F1  : 0.77
Profit F1: 0.94
Macro F1 : 0.85
ROC AUC  : 0.85+

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ml-pipeline.git
cd ml-pipeline
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the root directory:

### 4. Create PostgreSQL database
```sql
CREATE DATABASE ml_pipeline;
```

### 5. Run full pipeline (first time only)
```bash
python main.py
```

### 6. Launch dashboard (daily use)
```bash
streamlit run dashboard.py
```

---
## 🖥️ Dashboard Features

| Page | Description |
|---|---|
| 🏠 Home | Project overview and key metrics |
| 🔮 Prediction | Input campaign details → get Revenue and Profit/Loss prediction |
| 📈 Campaign Analytics | Brand, channel, and trend analysis with filters |
| 🤖 Model Performance | Metrics, confusion matrix, ROC curve |

---

## 🗄️ Database Schema

```sql
raw_campaigns                 -- original merged data
feature_engineered_campaigns  -- processed features
model_predictions             -- saved predictions
```

---

## 📌 Key Design Decisions

| Decision | Reason |
|---|---|
| RobustScaler over StandardScaler | Financial data has outliers |
| XGBoost over Random Forest | Better generalization, smaller overfit gap |
| Separate reg and clf pipelines | Different targets, different leakage rules |
| MLB for Channel_Used | Multi-value column — binary encoding per channel |
| VIF before voting | Remove redundant features before ranking |
| stratify=y in clf split | Preserve 77/23 class ratio in both splits |

---
