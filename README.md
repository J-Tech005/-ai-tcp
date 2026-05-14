# AI-Driven Test Case Prioritization (AI-TCP) System

**Supervisor:** Dr. Muhammad Aliyu  
**Student:** Jibrin Ali Jibrin (SFE/23U/3518)  
**Project:** Capstone / Final Year Project

---

## 📋 Overview

This system uses **machine learning to predict which regression tests are most likely to fail** and prioritizes them for early execution. By analyzing historical test execution patterns, the AI-TCP system:

1. **Trains predictive models** (Logistic Regression & Random Forest) on historical test data
2. **Scores each test** by failure probability and cost-aware risk
3. **Ranks tests** to maximize early fault detection
4. **Compares against baselines** (random and alphabetical orderings)
5. **Measures quality** using APFD, APFDc, TTFF, F1-score, and AUC-ROC
6. **Explains decisions** using SHAP feature importance
7. **Visualizes results** in an interactive Streamlit dashboard

---

## 🎯 Key Features

- **Synthetic Data Generation:** Realistic test execution logs (750 tests × 25 cycles)
- **Feature Engineering:** Failure rate, recency-weighted scores, change frequency, execution time
- **Dual ML Models:** Logistic Regression & Random Forest with hyperparameter tuning
- **Risk Scoring:** Standard and cost-aware variants
- **Evaluation Metrics:** APFD, APFDc, TTFF, Precision, Recall, F1, AUC-ROC
- **SHAP Explainability:** Per-prediction feature importance
- **Interactive Dashboard:** Streamlit web app with 6 pages
- **CLI Pipeline:** End-to-end command-line interface

---

## 🛠️ Prerequisites

- **Python 3.11+**
- **pip** (Python package manager)
- Standard laptop (no Docker, no cloud required)

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/J-Tech005/-ai-tcp.git
   cd -ai-tcp
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create required directories:**
   ```bash
   mkdir -p data/raw data/processed models results
   ```

---

## 🚀 Quick Start

### Step 1: Generate Synthetic Data
```bash
python scripts/generate_synthetic_data.py
```
Output: `data/raw/test_execution_history.csv` (18,750 records)

### Step 2: Run the Full Pipeline
```bash
python main.py --model rf --strategy risk --save-model
```

### Step 3: Launch the Interactive Dashboard
```bash
streamlit run dashboard/app.py
```
Access at: `http://localhost:8501`

### Step 4: Run Tests
```bash
pytest tests/ -v
```

---

## 📁 Project Structure

```
-ai-tcp/
├── data/
│   ├── raw/                        # Raw CSV files
│   └── processed/                  # Cleaned, feature-engineered data
│
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── data_ingestion.py           # DataIngestionLayer class
│   ├── preprocessing.py            # Data cleaning, encoding, scaling
│   ├── feature_engineering.py      # FeatureEngineer class
│   ├── ml_engine.py                # MLEngine abstract + concrete models
│   ├── risk_scorer.py              # RiskScorer class
│   ├── prioritization_engine.py    # PrioritizationEngine class
│   ├── evaluation.py               # EvaluationModule class
│   └── visualization.py            # Dashboard helpers
│
├── dashboard/
│   └── app.py                      # Streamlit app entry point
│
├── notebooks/
│   └── exploration.ipynb           # EDA and prototyping
│
├── models/
│   └── .gitkeep                    # Saved .joblib model files
│
├── tests/
│   ├── __init__.py
│   ├── test_data_ingestion.py
│   ├── test_feature_engineering.py
│   ├── test_risk_scorer.py
│   └── test_evaluation.py
│
├── scripts/
│   └── generate_synthetic_data.py  # Synthetic data generator
│
├── requirements.txt                # Python dependencies
├── main.py                         # CLI entry point
├── README.md                       # This file
└── .gitignore                      # Git ignore rules
```

---

## 📊 Metrics Explained

### APFD (Average Percentage of Faults Detected)
- **Formula:** `APFD = 1 - (Σ TF_i) / (n × m) + 1/(2n)`
- **Range:** [0, 1] (higher is better)
- **Meaning:** Proportion of faults detected as a function of tests executed
- **Best:** All failures detected first (APFD ≈ 1.0)
- **Worst:** All failures detected last (APFD ≈ 0.0)

### APFDc (Cost-Aware APFD)
- **Formula:** `APFDc = Σ(TF_i × (t_i - t_i/2)) / (TT × m)`
- **Range:** [0, 1] (higher is better)
- **Meaning:** APFD weighted by test execution time (penalizes slow tests)
- **Use Case:** When test execution time matters

### TTFF (Time To First Failure)
- **Unit:** Seconds
- **Meaning:** Cumulative execution time until the first failing test
- **Lower is better:** Catch bugs faster

### F1-Score
- **Formula:** `F1 = 2 × (Precision × Recall) / (Precision + Recall)`
- **Range:** [0, 1]
- **Meaning:** Harmonic mean of precision and recall

### AUC-ROC (Area Under the ROC Curve)
- **Range:** [0, 1]
- **Meaning:** Model's ability to distinguish failing vs. passing tests
- **0.5:** Random classifier, **1.0:** Perfect classifier

---

## 🎮 Dashboard Pages

1. **Upload & Run** — Upload CSV, run pipeline
2. **Ranked Test List** — Prioritized test order with scores
3. **Risk Score Distribution** — Histograms and risk breakdowns
4. **APFD / APFDc Curves** — Comparison vs. baselines
5. **Feature Importance (SHAP)** — Which features drive predictions
6. **Evaluation Report** — Classification metrics & comparisons

---

## 💻 CLI Commands

```bash
# Run with Random Forest, risk strategy
python main.py --model rf --strategy risk

# Run with Logistic Regression, APFDc strategy
python main.py --model lr --strategy apfdc

# Run with hyperparameter tuning and save model
python main.py --model rf --tune --save-model

# Specify input/output paths
python main.py --data data/raw/test_execution_history.csv \
               --output results/ranked_list.csv \
               --model rf
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data_ingestion.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 📈 Expected Results

When run on synthetic data:

```
=== AI-TCP Results ===
Model:          Random Forest (tuned)
APFD (AI):      0.847
APFD (Random):  0.512
APFD (Alpha):   0.498
APFDc (AI):     0.791
TTFF:           4.32s
F1 Score:       0.84
AUC-ROC:        0.91
Ranked list saved to: results/ranked_list.csv
```

AI-prioritized ranking should **significantly outperform** random and alphabetical baselines.

---

## 📝 Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11 | Core language |
| scikit-learn | 1.4.2 | ML models, metrics |
| pandas | 2.1.4 | Data manipulation |
| NumPy | 1.26.4 | Numerical computing |
| SHAP | 0.46.0 | Model explainability |
| Streamlit | 1.32.0 | Interactive dashboard |
| joblib | 1.3.2 | Model persistence |
| pytest | 8.1.1 | Unit testing |
| Matplotlib / Seaborn | latest | Visualization |
| Jupyter | latest | Notebooks |

---

## 🔬 Architecture

```
┌─────────────────────────┐
│  Raw Test Logs (CSV)    │
└────────────┬────────────┘
             │
             ▼
    ┌─────────────────┐
    │   Ingestion     │ ← DataIngestionLayer
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Preprocessing   │ ← clean_data, encode_labels, scale_features
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Feature Engr.   │ ← FeatureEngineer (failure_rate, recency_weighted)
    └────────┬────────┘
             │
             ▼
    ┌──────────────────┐
    │  Train/Eval      │ ← MLEngine (LR + RF)
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  Risk Scoring    │ ← RiskScorer
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Prioritization   │ ← PrioritizationEngine
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  Evaluation      │ ← EvaluationModule (APFD, APFDc, TTFF)
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  Visualization   │ ← Streamlit Dashboard + SHAP
    └──────────────────┘
```

---

## 📚 References

- Rothermel, G., Untch, R. H., Chu, C., & Harrold, M. J. (2001). Prioritizing test cases for regression testing. IEEE transactions on software engineering.
- Yoo, S., & Harman, M. (2012). Regression testing minimization, selection and prioritization. Software Testing, Verification and Reliability.
- Ahmed, A. B., Zamli, K. Z., & Lada, S. P. C. O. H. (2022). Adaptive cost-aware test case prioritization.

---

## 👤 Author

**Jibrin Ali Jibrin**  
Student ID: SFE/23U/3518  
Final Year, Software Engineering  
University: [Your University]  
Supervisor: Dr. Muhammad Aliyu

---

## 📄 License

This project is provided as-is for educational purposes.

---

## ❓ Support

For issues or questions, please open a GitHub issue or contact the author.
