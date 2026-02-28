# 🛡️ A25 - Cybersecurity Attack Type Detection
### Machine Learning Project | DSTI Applied MSc in Data Science & AI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Models](#-models)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Team](#-team)

---

## 🔍 Project Overview

This project aims to build a robust **multi-class classification system** to detect and categorize cybersecurity attack types from network traffic data. Using supervised machine learning and advanced preprocessing techniques, we classify attacks including **DDoS**, **Intrusion**, and **Malware** events.

> ⚠️ **Note:** Findings are preliminary and may evolve as the Data Engineering (DE) team completes data cleaning and feature encoding.

**Core Objectives:**
- Understand the structure and quality of raw cybersecurity network data
- Identify data quality issues and perform thorough preprocessing
- Form and validate feature importance hypotheses
- Train and compare multiple ML models for attack type classification
- Deliver a reliable, deployable classification pipeline

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Size** | ~40,000 rows |
| **Features** | 25 metrics (network/system attributes) |
| **Target** | Attack type (DDoS / Intrusion / Malware / Benign) |
| **State** | Raw → Cleaned via DE pipeline |

**Data files included:**
- `smote_ddos.csv` — SMOTE-augmented DDoS attack samples
- `smote_intrusion.csv` — SMOTE-augmented intrusion attack samples
- `smote_malware.csv` — SMOTE-augmented malware attack samples

> SMOTE (Synthetic Minority Over-sampling Technique) was applied to handle class imbalance.

---

## 📁 Project Structure

```
A25-Machine-Learning-project-DSTI/
│
├── 📓 Notebooks
│   ├── model_training_per_attack_type_for_report_v5i...  # Training pipeline v5
│   └── model_training_per_attack_type_for_report_v6i...  # Training pipeline v6
│
├── 📦 Models
│   └── rf_pca_model.pkl                                   # Saved Random Forest + PCA model
│
├── 📊 Data
│   ├── smote_ddos.csv
│   ├── smote_intrusion.csv
│   └── smote_malware.csv
│
├── ⚙️ Config
│   └── requirements.txt                                   # Python dependencies
│
└── README.md
```

---

## 🔬 Methodology

```
Raw Data
   │
   ▼
Exploratory Data Analysis (EDA)
   │  • Identify distributions, missing values, outliers
   │  • Preliminary feature importance
   │
   ▼
Preprocessing & Feature Engineering
   │  • Handling missing data
   │  • Encoding categorical variables
   │  • Normalization / Standardization
   │  • SMOTE for class balancing
   │
   ▼
Dimensionality Reduction
   │  • PCA (Principal Component Analysis)
   │
   ▼
Model Training & Evaluation
   │  • Per attack-type binary classifiers
   │  • Multi-class classification
   │  • Cross-validation & hyperparameter tuning
   │
   ▼
Model Persistence
      • Saved as rf_pca_model.pkl
```

---

## 🤖 Models

| Model | Description |
|-------|-------------|
| **Random Forest + PCA** | Primary model — dimensionality reduction + ensemble learning |
| Binary classifiers | One per attack type (DDoS, Intrusion, Malware) |

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC (per class)

---

## 📈 Results

> Results are being refined as the DE pipeline is finalized. Updated benchmarks will be published here upon completion.

---

## ⚙️ Installation

**Prerequisites:** Python 3.8+

```bash
# 1. Clone the repository
git clone https://github.com/swz-json/A25-Machine-Learning-project-DSTI.git
cd A25-Machine-Learning-project-DSTI

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

```python
import pickle
import pandas as pd

# Load the trained model
with open('rf_pca_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare your data (must match training feature set)
X_new = pd.read_csv('your_network_data.csv')

# Predict attack type
predictions = model.predict(X_new)
print(predictions)
```

Or run the notebooks directly in Jupyter:

```bash
jupyter notebook
```

---

## 👥 Team

**A25 Group - DSTI Nice Sophia **

| Names | Role | Responsibility |
|------|------|---------------|
|      | Data Engineering (DE) | Data cleaning, feature encoding, pipeline |
|      | Data Scientist (DS) | EDA, modeling, evaluation |
|      | Data Analyst (DA) | Reporting |

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

<div align="center">
  <sub>Built with ❤️ by the A25 team at DSTI</sub>
</div>
