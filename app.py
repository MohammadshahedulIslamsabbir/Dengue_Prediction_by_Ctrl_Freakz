
import numpy as np
import streamlit as st
import pandas as pd
from pathlib import Path

from utils import load_dengue_models, classify_dengue_risk


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Dengue Outburst Prediction | Ctrl Freakz",
    page_icon="🦟",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "processed" / "latest_ward_features.csv"
)

OUTPUT_PATH = BASE_DIR / "outputs"

PERFORMANCE_PATH = OUTPUT_PATH / "model_performance.csv"

MONTHLY_PATH = (
    OUTPUT_PATH / "monthly_actual_vs_predicted.csv"
)


# ==========================================
# 2. PROFESSIONAL WEBSITE DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #f4f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1 {
    color: #142d4e !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #183b63 !important;
    font-weight: 700 !important;
}

[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #e1e8f0;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(20, 45, 78, 0.05);
}

/* RED PREDICTION BUTTON */

.stButton > button {
    background-color: #dc2626 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    font-weight: 700;
}

.stButton > button:hover {
    background-color: #b91c1c !important;
    color: white !important;
    border: none !important;
}

section[data-testid="stSidebar"] {
    background-color: #142d4e;
}

section[data-testid="stSidebar"] * {
    color: white;
}

hr {
    border-color: #dce5ef;
}

[data-testid="stDataFrame"] {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. RISK BADGE AND TABLE COLOR FUNCTIONS
# ==========================================

def get_risk_badge(risk):

    colors = {
        "Low": ("#dcfce7", "#166534"),
        "Medium": ("#fef9c3", "#854d0e"),
        "High": ("#ffedd5", "#c2410c"),
        "Critical": ("#fee2e2", "#b91c1c")
    }

    background, text_color = colors[risk]

    return f"""
    <span style="
        background-color: {background};
        color: {text_color};
        padding: 7px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
    ">
        {risk.upper()}
    </span>
    """


def style_risk_table(data):

    colors = {
        "Low": "background-color: #dcfce7; color: #166534; font-weight: bold;",
        "Medium": "background-color: #fef9c3; color: #854d0e; font-weight: bold;",
        "High": "background-color: #ffedd5; color: #c2410c; font-weight: bold;",
        "Critical": "background-color: #fee2e2; color: #b91c1c; font-weight: bold;"
    }

    return data.style.map(
        lambda value: colors.get(value, ""),
        subset=["Risk Level"]
    )


# ==========================================
# 4. SIDEBAR BRANDING
# ==========================================

with st.sidebar:

    st.title("CTRL FREAKZ")

    st.caption("AI Dengue Monitoring System")

    st.divider()

    st.markdown("### Team Members")

    st.markdown("""
    **1. Mohammad Shahedul Islam Sabbir**

    **2. Washeka Rahman**

    **3. Samin Yeasar Islam**
    """)

    st.divider()

    st.write("Project: Dengue Outbreak Prediction")
    st.write("Forecast: 2–4 Weeks")
    st.write("Coverage: 5 Cities | 60 Wards")

# ==========================================
# 5. WEBSITE TITLE
# ==========================================

st.title("Dengue Outburst Prediction")

st.caption(
    "Team Ctrl Freakz | AI for Urban Dengue Outbreak"
)


# ==========================================
# 6. LOAD MODELS AND DATA
# ==========================================

@st.cache_resource
def get_models():

    return load_dengue_models()


@st.cache_data
def get_data():

    return pd.read_csv(
        DATA_PATH,
        parse_dates=["date"]
    )


@st.cache_data
def get_performance_data():

    return pd.read_csv(PERFORMANCE_PATH)


@st.cache_data
def get_monthly_data():

    return pd.read_csv(MONTHLY_PATH)


try:

    models = get_models()
    ward_data = get_data()

except Exception as e:

    st.error(f"Unable to load models or data: {e}")
    st.stop()


st.info(
    "Demo predictions use historical dataset features. "
    "Latest available input date: 1 December 2025. "
    "These are not live 2026 forecasts."
)


# ==========================================
# 7. MODEL FEATURES
# ==========================================

feature_columns = [

    "population_density_sqkm",
    "water_logging_risk_index",
    "sanitation_score",

    "rainfall_lag_1",
    "rainfall_lag_2",
    "rainfall_lag_3",
    "rainfall_lag_4",

    "rainfall_sum_2w",
    "rainfall_sum_3w",
    "rainfall_sum_4w",

    "avg_temp_lag_1",
    "avg_temp_lag_2",
    "avg_temp_lag_3",
    "avg_temp_lag_4",

    "max_temp_lag_1",
    "max_temp_lag_2",
    "max_temp_lag_3",
    "max_temp_lag_4",

    "humidity_lag_1",
    "humidity_lag_2",
    "humidity_lag_3",
    "humidity_lag_4",

    "dengue_lag_1",
    "dengue_lag_2",
    "dengue_lag_3",
    "dengue_lag_4",

    "month_sin",
    "month_cos",

    "city",
    "ward_id"
]


# ==========================================
# 8. DENGUE RISK PREDICTION
# ==========================================

st.divider()

st.header("Dengue Risk Prediction")

col1, col2 = st.columns(2)

with col1:

    selected_city = st.selectbox(
        "Select City",
        sorted(ward_data["city"].unique()),
        key="prediction_city"
    )


city_data = ward_data[
    ward_data["city"] == selected_city
]


with col2:

    selected_ward = st.selectbox(
        "Select Ward",
        sorted(city_data["ward_id"].unique()),
        key="prediction_ward"
    )


selected_horizon = st.selectbox(
    "Select Prediction Period",
    [2, 3, 4],
    format_func=lambda x: f"{x} Weeks Ahead"
)


if st.button("Predict Dengue Risk", type="primary"):

    selected_row = city_data[
        city_data["ward_id"] == selected_ward
    ]

    X = selected_row[feature_columns]

    # Predictions for all three horizons

    all_predictions = {

        "2 Weeks": models[2].predict(X)[0],

        "3 Weeks": models[3].predict(X)[0],

        "4 Weeks": models[4].predict(X)[0]

    }

    predicted_cases = all_predictions[
        f"{selected_horizon} Weeks"
    ]

    risk = classify_dengue_risk(predicted_cases)

    # Prediction result

    st.divider()

    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Estimated Dengue Cases (Model Output)",
            f"{predicted_cases:.4f}"
        )

    with result_col2:

        st.write("**Outbreak Risk Level**")

        st.markdown(
            get_risk_badge(risk),
            unsafe_allow_html=True
        )

    st.write(f"**City:** {selected_city}")

    st.write(f"**Ward:** {selected_ward}")

    st.write(
        f"**Forecast period:** {selected_horizon} weeks ahead"
    )

    st.write(
        f"**Input data date:** "
        f"{selected_row['date'].iloc[0].date()}"
    )

    # Prediction comparison chart

    st.divider()

    st.subheader(
        "Prediction Across Different Time Horizons"
    )

    chart_data = pd.DataFrame({

        "Prediction Period": list(all_predictions.keys()),

        "Estimated Cases": list(all_predictions.values())

    })

    st.bar_chart(
        chart_data,
        x="Prediction Period",
        y="Estimated Cases"
    )

    # Risk comparison table

    st.subheader("Risk Comparison")

    risk_data = pd.DataFrame({

        "Prediction Period": list(all_predictions.keys()),

        "Estimated Cases": [
            round(value, 4)
            for value in all_predictions.values()
        ],

        "Risk Level": [
            classify_dengue_risk(value)
            for value in all_predictions.values()
        ]

    })

    st.dataframe(
        style_risk_table(risk_data),
        hide_index=True,
        use_container_width=True
    )

    # ==========================================
    # 8.1 PREDICTION FACTORS AND EXPLANATION
    # ==========================================

    st.divider()

    st.subheader("Factors Behind Dengue Prediction")

    selected_model = models[selected_horizon]

    preprocessor = selected_model.named_steps["preprocessor"]

    rf_model = selected_model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    importance_data = pd.DataFrame({

        "Feature": feature_names,

        "Importance": rf_model.feature_importances_

    })

    top_features = (

        importance_data

        .sort_values("Importance", ascending=False)

        .head(5)

        .copy()

    )

    top_features["Feature"] = (

        top_features["Feature"]

        .str.replace("numeric__", "", regex=False)

        .str.replace("categorical__", "", regex=False)

        .str.replace("_", " ")

        .str.title()

    )

    st.write(
        "These are the five most important features "
        "used by the selected Random Forest model."
    )

    st.bar_chart(
        top_features.set_index("Feature")["Importance"]
    )

    # Selected ward conditions

    st.subheader("Selected Ward Conditions")

    ward_row = selected_row.iloc[0]

    condition1, condition2, condition3 = st.columns(3)

    condition1.metric(
        "Previous 2-Week Rainfall",
        f"{ward_row['rainfall_sum_2w']:.2f} mm"
    )

    condition2.metric(
        "Previous Week Temperature",
        f"{ward_row['avg_temp_lag_1']:.2f} °C"
    )

    condition3.metric(
        "Previous Week Humidity",
        f"{ward_row['humidity_lag_1']:.2f}%"
    )

    condition4, condition5 = st.columns(2)

    condition4.metric(
        "Waterlogging Risk Index",
        f"{ward_row['water_logging_risk_index']:.2f}"
    )

    condition5.metric(
        "Population Density",
        f"{ward_row['population_density_sqkm']:,.0f}"
    )

    st.caption(
        "Feature importance represents overall model influence, "
        "not proof of causation. Ward conditions are historical "
        "input measurements, not live weather readings."
    )


# ==========================================
# 9. DENGUE OUTBREAK OVERVIEW
# ==========================================

st.divider()

st.header("Dengue Outbreak Overview")

# Predict all wards using the 2-week model

X_all_wards = ward_data[feature_columns]

overview_predictions = models[2].predict(X_all_wards)


# ==========================================
# 9.0 MODEL PREDICTION DIAGNOSTICS
# ==========================================

st.subheader("Model Prediction Diagnostics")

prediction_summary = pd.Series(
    overview_predictions,
    name="Predicted Cases"
).describe()


diagnostic_col1, diagnostic_col2, diagnostic_col3 = st.columns(3)

diagnostic_col1.metric(
    "Minimum Prediction",
    f"{overview_predictions.min():.4f}"
)

diagnostic_col2.metric(
    "Maximum Prediction",
    f"{overview_predictions.max():.4f}"
)

diagnostic_col3.metric(
    "Average Prediction",
    f"{overview_predictions.mean():.4f}"
)


with st.expander("View Detailed Prediction Diagnostics"):

    st.write("Prediction Statistics")

    st.dataframe(prediction_summary)

    diagnostic_data = ward_data[
        ["city", "ward_id", "date"]
    ].copy()

    diagnostic_data["Predicted Cases"] = overview_predictions

    diagnostic_data = diagnostic_data.sort_values(
        "Predicted Cases",
        ascending=False
    )

    st.write("All 60 Ward Predictions")

    st.dataframe(
        diagnostic_data,
        use_container_width=True,
        hide_index=True
    )

    st.write(
        "Predictions below zero:",
        int((overview_predictions < 0).sum())
    )

    st.write(
        "Predictions below 0.01:",
        int((overview_predictions < 0.01).sum())
    )

    st.write(
        "All predictions are finite:",
        bool(np.isfinite(overview_predictions).all())
    )


overview_data = ward_data[
    ["city", "ward_id", "date"]
].copy()

overview_data["Predicted Cases"] = overview_predictions

overview_data["Risk Level"] = [

    classify_dengue_risk(value)

    for value in overview_predictions

]


# ==========================================
# 9.1 RED ALERT - CRITICAL RISK WARDS
# ==========================================

st.divider()

st.subheader("🔴 Red Alert - Critical Risk Wards")

critical_wards = overview_data[

    overview_data["Risk Level"] == "Critical"

].copy()


critical_wards = critical_wards.sort_values(
    "Predicted Cases",
    ascending=False
)

critical_count = len(critical_wards)

st.metric(
    "Total Critical Risk Wards",
    critical_count
)


if not critical_wards.empty:

    critical_display = critical_wards[

        ["city", "ward_id", "Predicted Cases", "Risk Level"]

    ].copy()

    critical_display["Predicted Cases"] = (
        critical_display["Predicted Cases"].round(4)
    )

    st.error(
        f"{critical_count} wards are classified as Critical Risk "
        "by the 2-week prediction model."
    )

    st.dataframe(
        style_risk_table(critical_display),
        use_container_width=True,
        hide_index=True
    )

else:

    st.success(
        "No wards are currently classified as Critical Risk "
        "by the 2-week demonstration model."
    )


# ==========================================
# 9.2 DASHBOARD STATISTICS
# ==========================================

total_wards = overview_data["ward_id"].nunique()

total_cities = overview_data["city"].nunique()

high_risk_wards = overview_data[

    overview_data["Risk Level"].isin(["High", "Critical"])

]["ward_id"].nunique()

avg_prediction = overview_data["Predicted Cases"].mean()


metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric("Total Wards", total_wards)

metric2.metric("Total Cities", total_cities)

metric3.metric("High/Critical Risk Wards", high_risk_wards)

metric4.metric(
    "Average Estimated Cases",
    f"{avg_prediction:.4f}"
)


st.caption(
    "Overview uses 2-week model predictions based on "
    "historical input data dated 1 December 2025. "
    "These are demonstration estimates, not live outbreak statistics."
)


# ==========================================
# 9.3 WARD RISK DISTRIBUTION
# ==========================================

st.divider()

st.subheader("Ward Risk Distribution")

risk_order = ["Low", "Medium", "High", "Critical"]

risk_counts = (

    overview_data["Risk Level"]

    .value_counts()

    .reindex(risk_order, fill_value=0)

)

risk_chart_data = pd.DataFrame({

    "Risk Level": risk_counts.index,

    "Number of Wards": risk_counts.values

})

st.bar_chart(
    risk_chart_data,
    x="Risk Level",
    y="Number of Wards"
)


# ==========================================
# 9.4 WARD-LEVEL RISK SUMMARY
# ==========================================

st.divider()

st.subheader("Ward-Level Risk Summary")

selected_overview_city = st.selectbox(

    "Filter Dashboard by City",

    ["All Cities"] + sorted(overview_data["city"].unique()),

    key="overview_city"

)


if selected_overview_city != "All Cities":

    display_data = overview_data[

        overview_data["city"] == selected_overview_city

    ]

else:

    display_data = overview_data.copy()


display_data = display_data.sort_values(
    "Predicted Cases",
    ascending=False
)

st.dataframe(
    style_risk_table(display_data),
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 10. MODEL PERFORMANCE ANALYSIS
# ==========================================

st.divider()

st.header("Model Performance Analysis")

st.write(
    "Evaluate how accurately the three Random Forest models "
    "predicted dengue cases using the historical 2025 test dataset."
)


# Load evaluation CSV files

try:

    performance_data = get_performance_data()

    monthly_data = get_monthly_data()

except Exception as e:

    st.error(
        "Unable to load model evaluation files. "
        "Please check that both CSV files exist inside the outputs folder."
    )

    st.code(str(e))

    st.stop()


# ------------------------------------------
# 10.1 SELECT EVALUATION PERIOD
# ------------------------------------------

selected_period = st.selectbox(

    "Select Evaluation Period",

    ["Overall Test", "August 2025"],

    key="performance_period"

)


period_data = performance_data[

    performance_data["Period"] == selected_period

].copy()


period_data = period_data.sort_values(
    "Forecast Horizon"
)


st.subheader(f"{selected_period} Performance")


# ------------------------------------------
# 10.2 PERFORMANCE METRICS
# ------------------------------------------

performance_col1, performance_col2, performance_col3 = st.columns(3)


for index, horizon in enumerate([2, 3, 4]):

    horizon_name = f"{horizon} Weeks"

    result = period_data[

        period_data["Forecast Horizon"] == horizon_name

    ]

    if not result.empty:

        row = result.iloc[0]

        if index == 0:

            performance_col1.metric(
                "2-Week MAE",
                f"{row['MAE']:.4f}"
            )

            performance_col1.caption(
                f"RMSE: {row['RMSE']:.4f}"
            )

        elif index == 1:

            performance_col2.metric(
                "3-Week MAE",
                f"{row['MAE']:.4f}"
            )

            performance_col2.caption(
                f"RMSE: {row['RMSE']:.4f}"
            )

        else:

            performance_col3.metric(
                "4-Week MAE",
                f"{row['MAE']:.4f}"
            )

            performance_col3.caption(
                f"RMSE: {row['RMSE']:.4f}"
            )


# ------------------------------------------
# 10.3 MAE COMPARISON
# ------------------------------------------

st.divider()

st.subheader("Mean Absolute Error (MAE) Comparison")

st.write(
    "MAE measures the average absolute difference "
    "between actual and predicted dengue cases. "
    "Lower MAE indicates better performance."
)


mae_chart_data = period_data[

    ["Forecast Horizon", "MAE"]

].set_index("Forecast Horizon")


st.bar_chart(mae_chart_data)


# ------------------------------------------
# 10.4 RMSE COMPARISON
# ------------------------------------------

st.subheader("Root Mean Squared Error (RMSE) Comparison")

st.write(
    "RMSE measures prediction errors while giving "
    "larger errors more weight. Lower RMSE is better."
)


rmse_chart_data = period_data[

    ["Forecast Horizon", "RMSE"]

].set_index("Forecast Horizon")


st.bar_chart(rmse_chart_data)


# ------------------------------------------
# 10.5 PERFORMANCE TABLE
# ------------------------------------------

st.subheader("Detailed Model Performance")

performance_display = period_data.copy()

performance_display["MAE"] = (
    performance_display["MAE"].round(4)
)

performance_display["RMSE"] = (
    performance_display["RMSE"].round(4)
)

st.dataframe(
    performance_display,
    use_container_width=True,
    hide_index=True
)


# ------------------------------------------
# 10.6 MONTHLY ACTUAL VS PREDICTED
# ------------------------------------------

st.divider()

st.subheader("Monthly Actual vs Predicted Dengue Cases")

st.write(
    "Compare the average actual dengue cases with "
    "the average model predictions for each month "
    "of the 2025 test period."
)


selected_monthly_horizon = st.selectbox(

    "Select Forecast Horizon for Monthly Analysis",

    ["2 Weeks", "3 Weeks", "4 Weeks"],

    key="monthly_horizon"

)


monthly_display = monthly_data[

    monthly_data["Forecast Horizon"] == selected_monthly_horizon

].copy()


# Keep months in calendar order

month_order = [

    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"

]


monthly_display["Month"] = pd.Categorical(

    monthly_display["Month"],

    categories=month_order,

    ordered=True

)


monthly_display = monthly_display.sort_values("Month")


# Create chart

monthly_chart_data = monthly_display.set_index("Month")[

    ["Actual_Average", "Predicted_Average"]

].rename(columns={

    "Actual_Average": "Actual Cases",

    "Predicted_Average": "Predicted Cases"

})


st.line_chart(monthly_chart_data)


# Monthly comparison table

monthly_table = monthly_display.copy()

monthly_table["Actual_Average"] = (
    monthly_table["Actual_Average"].round(4)
)

monthly_table["Predicted_Average"] = (
    monthly_table["Predicted_Average"].round(4)
)

monthly_table = monthly_table.rename(columns={

    "Actual_Average": "Actual Average Cases",

    "Predicted_Average": "Predicted Average Cases",

    "Record_Count": "Number of Records"

})


st.subheader("Monthly Performance Data")

st.dataframe(
    monthly_table,
    use_container_width=True,
    hide_index=True
)


# ------------------------------------------
# 10.7 BEST MODEL INDICATOR
# ------------------------------------------

st.divider()

st.subheader("Best Performing Model")


best_model_row = period_data.loc[
    period_data["MAE"].idxmin()
]


st.success(
    f"The {best_model_row['Forecast Horizon']} model "
    f"has the lowest MAE ({best_model_row['MAE']:.4f}) "
    f"for the {selected_period} evaluation period."
)


st.caption(
    "Evaluation results are calculated on historical test data. "
    "They do not guarantee future prediction accuracy. "
    "August results represent a seasonal subset of the 2025 test data."
)


# ==========================================
# 11. FOOTER
# ==========================================

st.divider()

st.caption(
    "Developed by Team Ctrl Freakz | "
    "AI for Urban Dengue Outbreak"
)
