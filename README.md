# Dengue Outburst Prediction

**Team Name:** CTRL FREAKZ

**Team Members:**
1. Mohammad Shahedul Islam Sabbir
2. Washeka Rahman
3. Samin Yeasar Islam

## 1. Project Overview

Dengue outbreaks are a major public health concern in Bangladesh. This project aims to provide an early-warning system that predicts dengue outbreak risks at the ward level.

The system uses historical dengue cases, weather conditions, and ward vulnerability information to estimate dengue cases 2, 3, and 4 weeks ahead.

It helps public health officers identify potentially high-risk wards earlier and supports better planning for dengue prevention and resource allocation.

## 2. Key Features

- Ward-level dengue prediction for 2, 3, and 4 weeks ahead.
- Rainfall lag and multi-week rainfall feature engineering.
- Temperature and humidity-based features.
- Ward vulnerability analysis.
- Three trained Random Forest regression models.
- Four risk categories: Low, Medium, High, and Critical.
- Interactive Streamlit dashboard.
- Model performance evaluation using MAE and RMSE.
- Monthly actual-versus-predicted comparison.
- Local model loading to reduce repeated computation and unnecessary API usage.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Matplotlib / Seaborn | Data visualization |
| Joblib | Model serialization |
| Streamlit | Interactive web dashboard |
| Jupyter Notebook | Data analysis and model development |

## 4. Dataset

The project uses the Bangladesh dengue dataset provided for the hackathon.

Dataset source:

[Hackathon Dengue Dataset](https://docs.google.com/spreadsheets/d/1QFI0ACJRvJQkiQT3LfGHklM3g4PgaJWUt8xSVU07NI/edit?usp=sharing)

The dataset contains historical weekly records from five cities:

- Chattogram
- Dhaka North
- Dhaka South
- Gazipur
- Narayanganj

It includes weather measurements, ward vulnerability indicators, and dengue case counts.

## 5. Machine Learning Approach

Three separate Random Forest Regression models were trained to predict dengue cases at different future time horizons:

- 2-week prediction model
- 3-week prediction model
- 4-week prediction model

### Feature Engineering

The pipeline creates:

- Rainfall lags for previous 1–4 weeks.
- Rolling rainfall sums for previous 2, 3, and 4 weeks.
- Temperature lags.
- Humidity lags.
- Previous dengue case lags.
- Monthly cyclical features.
- Ward vulnerability features.

### Risk Classification

The model's predicted case values are classified using project-defined thresholds:

| Predicted Cases | Risk Level |
|---|---|
| Below 1 | Low |
| 1–4 | Medium |
| Above 4–10 | High |
| Above 10 | Critical |

These are project-specific thresholds, not official government or WHO alert thresholds.

## 6. Model Evaluation

The models were evaluated using a chronological data split to avoid using future observations during training.

| Forecast Horizon | Validation MAE | Test MAE | Validation RMSE | Test RMSE |
|---|---:|---:|---:|---:|
| 2 Weeks | 1.1137 | 1.3849 | 2.0538 | 2.5800 |
| 3 Weeks | 1.3498 | 1.5697 | 2.4095 | 2.7012 |
| 4 Weeks | 1.4402 | 1.6282 | 2.5153 | 3.0125 |

Lower MAE and RMSE values indicate better prediction performance.

The 2-week model achieved the lowest test MAE and RMSE among the three models.

## 7. Project Structure

```text
Dengue_Prediction_by_Ctrl_Freakz/
│
├── app/
│   ├── app.py
│   └── utils.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│
├── models/
│   ├── dengue_model_2_weeks.pkl
│   ├── dengue_model_3_weeks.pkl
│   └── dengue_model_4_weeks.pkl
│
├── outputs/
│   ├── model_performance.csv
│   ├── monthly_actual_vs_predicted.csv
│   └── validation_test_performance.csv
│
├── docs/
│
├── README.md
├── ARCHITECTURE.md
├── requirements.txt
└── .gitignore
```

## 8. Installation and Running Instructions

### Requirements

- Python 3.10 or newer (use a compatible Python version for the project's dependencies).
- Git.
- Internet access for initial dependency installation.

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

```bash
cd Dengue_Prediction_by_Ctrl_Freakz
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 3: Activate the Environment

**Windows:**

```bash
.venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
python -m streamlit run app/app.py
```

The application will open in your browser, normally at:

```text
http://localhost:8501
```

## 9. Important Notes and Limitations

- The current prototype uses historical dataset features.
- The latest available input date in the demonstration data is 1 December 2025.
- Predictions are not live 2026 forecasts.
- Model predictions are estimates and should not replace public health authority decisions.
- Risk thresholds are defined for this project and are not official medical alert thresholds.
- Feature importance indicates model influence, not proof of causation.

## 10. AI Assistant and External API Disclosure

**AI Assistant Used:**
- ChatGPT by OpenAI — used for programming assistance, debugging, explanations, and documentation support.

**External APIs:**
- No external weather API or LLM API is used by the current prediction application.
- Predictions are generated locally using saved machine learning models.
- The original hackathon dataset is used as the data source.

Update this disclosure if any other AI assistants or external APIs were used by the team.

## 11. Team

**CTRL FREAKZ**

- Mohammad Shahedul Islam Sabbir
- Washeka Rahman
- Samin Yeasar Islam

**Challenge:**  Dengue Outburst Prediction
**Hackathon:** Preliminary Round — Phase 1
