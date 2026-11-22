# Dengue Outbreak Prediction
**Team Name: CTRL FREAKZ**

**Challenge:** Dengue Outbreak Prediction  
**Hackathon:** Preliminary Round – Phase 1

## 1. Project Overview
This project is an early-warning system that predicts ward-level dengue cases in Bangladesh 2, 3, and 4 weeks ahead.

It uses historical dengue cases, weather conditions, and ward vulnerability data to identify potential high-risk areas and support early prevention planning.

## 2. Key Features
- Ward-level dengue prediction for 2, 3, and 4 weeks ahead.
- Rainfall, temperature, and humidity feature engineering.
- Ward vulnerability analysis.
- Three Random Forest regression models.
- Four risk levels: Low, Medium, High, and Critical.
- Interactive Streamlit dashboard.
- Model evaluation using MAE and RMSE.
- Monthly actual-versus-predicted comparison.
- Local model loading for efficient predictions.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Pandas, NumPy | Data processing |
| Scikit-learn | Machine learning |
| Matplotlib, Seaborn | Visualization |
| Joblib | Model serialization |
| Streamlit | Dashboard |
| Jupyter Notebook | Model development |

## 4. Dataset
**Source:** Hackathon Dengue Dataset

The dataset contains historical weekly dengue records from five cities:
- Chattogram
- Dhaka North
- Dhaka South
- Gazipur
- Narayanganj

It includes weather data, ward vulnerability indicators, and dengue case counts.

## 5. Machine Learning Approach

Three separate Random Forest Regression models predict dengue cases:

- 2-week forecast
- 3-week forecast
- 4-week forecast

**Feature Engineering:**
- Rainfall lags (1–4 weeks).
- Rolling rainfall sums (2–4 weeks).
- Temperature and humidity lags.
- Previous dengue case lags.
- Monthly cyclical features.
- Ward vulnerability features.

**Risk Classification:**

| Predicted Cases | Risk Level |
|---|---|
| Below 1 | Low |
| 1–4 | Medium |
| Above 4–10 | High |
| Above 10 | Critical |

*These are project-defined thresholds, not official government or WHO alert levels.*

## 6. Model Evaluation

Models were evaluated using chronological data splitting.

| Forecast | Validation MAE | Test MAE | Validation RMSE | Test RMSE |
|---|---:|---:|---:|---:|
| 2 Weeks | 1.1137 | 1.3849 | 2.0538 | 2.5800 |
| 3 Weeks | 1.3498 | 1.5697 | 2.4095 | 2.7012 |
| 4 Weeks | 1.4402 | 1.6282 | 2.5153 | 3.0125 |

The 2-week model achieved the lowest test MAE and RMSE.

## 7. Project Structure

```text
Dengue_Prediction_by_Ctrl_Freakz/
│
├── app/
│   ├── app.py
│   └── utils.py
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── models/
│   ├── dengue_model_2_weeks.pkl
│   ├── dengue_model_3_weeks.pkl
│   └── dengue_model_4_weeks.pkl
├── outputs/
├── docs/
├── README.md
├── ARCHITECTURE.md
├── requirements.txt
└── .gitignore
```

## 8. Installation and Execution

**Requirements:** Python 3.10+, Git, and internet access for initial dependency installation.

**1. Clone the repository**
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Dengue_Prediction_by_Ctrl_Freakz
```

**2. Create and activate virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**3. Install dependencies**
```bash
python -m pip install -r requirements.txt
```

**4. Run the application**
```bash
python -m streamlit run app/app.py
```

Open: `http://localhost:8501`

## 9. Limitations
- Uses historical dataset features.
- Latest demonstration data date: 1 December 2025.
- Does not provide live 2026 forecasts.
- Predictions are estimates, not substitutes for public health decisions.
- Risk thresholds are project-defined.
- Feature importance does not establish causation.

## 10. AI and External API Disclosure

**AI Assistant:** ChatGPT by OpenAI — used for coding assistance, debugging, explanations, and documentation.

**External APIs:** No external weather or LLM APIs are used. Predictions are generated locally using saved machine learning models and the hackathon dataset.

## 11. Team CTRL FREAKZ

1. Mohammad Shahedul Islam Sabbir
2. Washeka Rahman
3. Samin Yeasar Islam
