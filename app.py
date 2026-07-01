import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================
# Page setup
# =============================
st.set_page_config(
    page_title="Heart Risk Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = "heart_disease_risk_dataset_earlymed.csv"
TARGET = "Heart_Risk"

BASE_FEATURES = [
    "Chest_Pain",
    "Shortness_of_Breath",
    "Fatigue",
    "Palpitations",
    "Dizziness",
    "Swelling",
    "Pain_Arms_Jaw_Back",
    "Cold_Sweats_Nausea",
    "High_BP",
    "High_Cholesterol",
    "Diabetes",
    "Smoking",
    "Obesity",
    "Sedentary_Lifestyle",
    "Family_History",
    "Chronic_Stress",
    "Gender",
    "Age",
]

SYMPTOM_FEATURES = [
    "Chest_Pain",
    "Shortness_of_Breath",
    "Fatigue",
    "Palpitations",
    "Dizziness",
    "Swelling",
    "Pain_Arms_Jaw_Back",
    "Cold_Sweats_Nausea",
]

RISK_SYMPTOM_FEATURES = [
    "Chest_Pain",
    "Shortness_of_Breath",
    "Pain_Arms_Jaw_Back",
    "Cold_Sweats_Nausea",
    "Dizziness",
]

LIFESTYLE_FEATURES = [
    "Smoking",
    "Obesity",
    "Sedentary_Lifestyle",
    "Chronic_Stress",
]

CHART_COLORS = {
    "background": "rgba(0,0,0,0)",
    "panel": "rgba(15, 23, 42, 0.72)",
    "grid": "rgba(148, 163, 184, 0.18)",
    "text": "#e5e7eb",
    "title": "#f8fafc",
    "teal": "#14b8a6",
    "blue": "#38bdf8",
    "orange": "#f97316",
    "red": "#fb7185",
    "green": "#22c55e",
    "purple": "#a78bfa",
}

CLASS_COLORS = {
    "No Risk": CHART_COLORS["teal"],
    "Risk": CHART_COLORS["orange"],
}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(20,184,166,0.18), transparent 30%),
            radial-gradient(circle at top right, rgba(56,189,248,0.14), transparent 28%),
            linear-gradient(135deg, #020617 0%, #0f172a 45%, #111827 100%);
        color: #e5e7eb;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617 0%, #0f172a 100%);
        border-right: 1px solid rgba(148,163,184,0.22);
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    .main-title {
        padding: 1.6rem 1.8rem;
        border-radius: 26px;
        background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(30,41,59,0.82));
        border: 1px solid rgba(148,163,184,0.22);
        box-shadow: 0 22px 50px rgba(0,0,0,0.24);
        margin-bottom: 1.2rem;
    }

    .main-title h1 {
        color: #f8fafc;
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 0.45rem;
    }

    .main-title p {
        color: #cbd5e1;
        font-size: 1.05rem;
        margin: 0;
    }

    .section-title {
        color: #f8fafc;
        font-size: 1.65rem;
        font-weight: 800;
        margin: 0.4rem 0 1rem 0;
    }

    .card {
        background: rgba(15,23,42,0.78);
        border: 1px solid rgba(148,163,184,0.20);
        border-radius: 24px;
        padding: 1.25rem;
        box-shadow: 0 18px 45px rgba(0,0,0,0.22);
        margin-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(145deg, rgba(15,23,42,0.94), rgba(30,41,59,0.72));
        border: 1px solid rgba(148,163,184,0.20);
        border-radius: 22px;
        padding: 1.15rem;
        min-height: 132px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.20);
    }

    .metric-card .label {
        color: #94a3b8;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }

    .metric-card .value {
        color: #f8fafc;
        font-size: 1.85rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }

    .metric-card .note {
        color: #cbd5e1;
        font-size: 0.84rem;
    }

    .result-low,
    .result-medium,
    .result-high {
        border-radius: 24px;
        padding: 1.25rem;
        margin-top: 0.4rem;
        border: 1px solid rgba(255,255,255,0.12);
    }

    .result-low {
        background: rgba(20,184,166,0.16);
    }

    .result-medium {
        background: rgba(249,115,22,0.18);
    }

    .result-high {
        background: rgba(251,113,133,0.18);
    }

    .result-title {
        color: #f8fafc;
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }

    .result-text {
        color: #cbd5e1;
        font-size: 0.98rem;
        margin: 0;
    }

    .predicted-class-card {
        margin-top: 1.1rem;
        padding: 1.35rem 1rem;
        border-radius: 24px;
        background: rgba(2, 6, 23, 0.62);
        border: 1px solid rgba(255, 255, 255, 0.16);
        text-align: center;
        box-shadow: 0 16px 36px rgba(0, 0, 0, 0.22);
    }

    .predicted-class-label {
        color: #cbd5e1;
        font-size: 1rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .predicted-class-value {
        font-size: 3.2rem;
        font-weight: 900;
        line-height: 1.05;
        margin: 0;
    }

    .predicted-no-risk {
        color: #22c55e;
        text-shadow: 0 0 18px rgba(34, 197, 94, 0.38);
    }

    .predicted-risk {
        color: #fb7185;
        text-shadow: 0 0 18px rgba(251, 113, 133, 0.38);
    }

    .small-note {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
    }

    .stButton>button {
        width: 100%;
        border-radius: 16px;
        border: 0;
        background: linear-gradient(135deg, #14b8a6, #38bdf8);
        color: #020617;
        font-weight: 800;
        padding: 0.85rem 1rem;
        font-size: 1rem;
    }

    .stButton>button:hover {
        border: 0;
        color: #020617;
        filter: brightness(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================
# Data and model functions
# =============================
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(DATA_FILE)

    data = pd.read_csv(DATA_FILE)
    return data


def validate_columns(data: pd.DataFrame) -> list[str]:
    required_columns = BASE_FEATURES + [TARGET]
    return [col for col in required_columns if col not in data.columns]


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["symptom"] = data[SYMPTOM_FEATURES].sum(axis=1)
    data["risk_symptom"] = data[RISK_SYMPTOM_FEATURES].sum(axis=1)
    data["lifestyle"] = data[LIFESTYLE_FEATURES].sum(axis=1)
    return data


@st.cache_data(show_spinner=False)
def clean_and_engineer(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    missing_columns = validate_columns(data)
    if missing_columns:
        raise ValueError("Missing columns: " + ", ".join(missing_columns))

    data = data[BASE_FEATURES + [TARGET]]
    data = data.dropna()
    data = data.astype(int)
    data = data.drop_duplicates()
    data = add_engineered_features(data)
    return data


@st.cache_resource(show_spinner=False)
def train_models(data: pd.DataFrame) -> dict:
    feature_columns = [col for col in data.columns if col != TARGET]
    X = data[feature_columns]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.95)),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.95)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=8,
                        min_samples_split=20,
                        min_samples_leaf=10,
                        max_features="sqrt",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    rows = []
    matrices = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred),
            }
        )
        matrices[model_name] = confusion_matrix(y_test, y_pred)

    metrics = pd.DataFrame(rows)
    pca_step = models["Logistic Regression"].named_steps["pca"]

    return {
        "models": models,
        "metrics": metrics,
        "confusion_matrices": matrices,
        "feature_columns": feature_columns,
        "pca_components": int(pca_step.n_components_),
        "explained_variance": float(pca_step.explained_variance_ratio_.sum()),
    }


# =============================
# UI helper functions
# =============================
def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_plot_layout(fig: go.Figure, height: int = 430) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CHART_COLORS["background"],
        plot_bgcolor=CHART_COLORS["panel"],
        font=dict(color=CHART_COLORS["text"], size=13),
        title_font=dict(color=CHART_COLORS["title"], size=22),
        xaxis=dict(gridcolor=CHART_COLORS["grid"], zerolinecolor=CHART_COLORS["grid"]),
        yaxis=dict(gridcolor=CHART_COLORS["grid"], zerolinecolor=CHART_COLORS["grid"]),
        margin=dict(l=20, r=20, t=70, b=30),
        height=height,
    )
    return fig


def binary_label(value: int) -> str:
    return "Risk" if int(value) == 1 else "No Risk"


def make_risk_distribution_chart(data: pd.DataFrame) -> go.Figure:
    chart_data = data[TARGET].map({0: "No Risk", 1: "Risk"}).value_counts().reset_index()
    chart_data.columns = ["Heart Risk", "Count"]

    fig = px.bar(
        chart_data,
        x="Heart Risk",
        y="Count",
        color="Heart Risk",
        text="Count",
        title="Heart Risk Distribution",
        color_discrete_map=CLASS_COLORS,
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(showlegend=False)
    return apply_plot_layout(fig)


def make_age_chart(data: pd.DataFrame) -> go.Figure:
    chart_data = data.copy()
    chart_data["Heart Risk"] = chart_data[TARGET].map({0: "No Risk", 1: "Risk"})

    fig = px.histogram(
        chart_data,
        x="Age",
        color="Heart Risk",
        nbins=32,
        barmode="overlay",
        opacity=0.72,
        title="Age Distribution by Heart Risk",
        color_discrete_map=CLASS_COLORS,
    )
    return apply_plot_layout(fig)


def make_engineered_features_chart(data: pd.DataFrame) -> go.Figure:
    chart_data = data.copy()
    chart_data["Heart Risk"] = chart_data[TARGET].map({0: "No Risk", 1: "Risk"})
    grouped = (
        chart_data.groupby("Heart Risk")[["symptom", "risk_symptom", "lifestyle"]]
        .mean()
        .reset_index()
        .melt(id_vars="Heart Risk", var_name="Feature", value_name="Average Score")
    )
    grouped["Feature"] = grouped["Feature"].replace(
        {
            "symptom": "Total Symptoms",
            "risk_symptom": "Risk Symptoms",
            "lifestyle": "Lifestyle Score",
        }
    )

    fig = px.bar(
        grouped,
        x="Feature",
        y="Average Score",
        color="Heart Risk",
        barmode="group",
        title="Average Engineered Feature Scores by Heart Risk",
        color_discrete_map=CLASS_COLORS,
    )
    return apply_plot_layout(fig)


def make_correlation_chart(data: pd.DataFrame) -> go.Figure:
    corr_columns = ["Age", "symptom", "risk_symptom", "lifestyle", TARGET]
    corr = data[corr_columns].corr()
    labels = {
        "Age": "Age",
        "symptom": "Total Symptoms",
        "risk_symptom": "Risk Symptoms",
        "lifestyle": "Lifestyle Score",
        TARGET: "Heart Risk",
    }
    corr = corr.rename(index=labels, columns=labels)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=["#0f172a", "#38bdf8", "#22c55e"],
        title="Correlation Heatmap Without Gender",
        aspect="auto",
    )
    fig.update_coloraxes(showscale=True)
    return apply_plot_layout(fig, height=480)


def make_model_comparison_chart(metrics: pd.DataFrame) -> go.Figure:
    chart_data = metrics.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
        var_name="Metric",
        value_name="Score",
    )
    chart_data["Score"] = chart_data["Score"] * 100

    fig = px.bar(
        chart_data,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text=chart_data["Score"].map(lambda x: f"{x:.2f}%"),
        title="Model Comparison: Logistic Regression vs Random Forest",
        color_discrete_map={
            "Logistic Regression": CHART_COLORS["blue"],
            "Random Forest": CHART_COLORS["purple"],
        },
    )
    fig.update_traces(textposition="outside")
    fig.update_yaxes(range=[95, 100])
    return apply_plot_layout(fig, height=500)


def make_confusion_matrix_chart(matrix: np.ndarray, title: str) -> go.Figure:
    fig = px.imshow(
        matrix,
        text_auto=True,
        x=["Predicted No Risk", "Predicted Risk"],
        y=["Actual No Risk", "Actual Risk"],
        color_continuous_scale=["#0f172a", "#38bdf8", "#22c55e"],
        title=title,
        aspect="auto",
    )
    return apply_plot_layout(fig, height=420)


def prediction_status(probability_percent: float) -> tuple[str, str, str]:
    if probability_percent >= 65:
        return (
            "High Risk",
            "result-high",
            "The selected model predicts a high heart disease risk. This is not a medical diagnosis.",
        )
    if probability_percent >= 35:
        return (
            "Medium Risk",
            "result-medium",
            "The selected model predicts a medium heart disease risk. Keep the result as an educational estimate.",
        )
    return (
        "Low Risk",
        "result-low",
        "The selected model predicts a low heart disease risk based on the selected answers.",
    )


def make_gauge(probability_percent: float) -> go.Figure:
    if probability_percent >= 65:
        gauge_color = CHART_COLORS["red"]
    elif probability_percent >= 35:
        gauge_color = CHART_COLORS["orange"]
    else:
        gauge_color = CHART_COLORS["teal"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability_percent,
            number={"suffix": "%", "font": {"size": 48, "color": "#f8fafc"}},
            title={"text": "Risk Percentage", "font": {"size": 20, "color": "#f8fafc"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#cbd5e1"},
                "bar": {"color": gauge_color},
                "bgcolor": "rgba(15,23,42,0.45)",
                "borderwidth": 1,
                "bordercolor": "rgba(148,163,184,0.25)",
                "steps": [
                    {"range": [0, 35], "color": "rgba(20,184,166,0.25)"},
                    {"range": [35, 65], "color": "rgba(249,115,22,0.25)"},
                    {"range": [65, 100], "color": "rgba(251,113,133,0.25)"},
                ],
                "threshold": {
                    "line": {"color": "#f8fafc", "width": 4},
                    "thickness": 0.75,
                    "value": probability_percent,
                },
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=10, r=10, t=60, b=10),
        font=dict(color="#e5e7eb"),
    )
    return fig


def yes_no_input(label: str, key: str) -> int:
    return 1 if st.toggle(label, key=key) else 0


def build_patient_input() -> pd.DataFrame:
    st.markdown("### Patient Information")
    info_1, info_2 = st.columns(2)
    with info_1:
        age = st.slider("Age", min_value=18, max_value=90, value=45, step=1)
    with info_2:
        gender_text = st.selectbox("Gender", ["Female", "Male"])
        gender = 0 if gender_text == "Female" else 1

    st.markdown(
        "<div class='small-note'>Gender is available in the prediction form only. It is not used to filter or change the dashboard charts.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### Symptoms")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        chest_pain = yes_no_input("Chest pain", "chest_pain")
        shortness = yes_no_input("Shortness of breath", "shortness")
    with s2:
        fatigue = yes_no_input("Fatigue", "fatigue")
        palpitations = yes_no_input("Palpitations", "palpitations")
    with s3:
        dizziness = yes_no_input("Dizziness", "dizziness")
        swelling = yes_no_input("Swelling", "swelling")
    with s4:
        arms_pain = yes_no_input("Pain in arms / jaw / back", "arms_pain")
        cold_sweats = yes_no_input("Cold sweats / nausea", "cold_sweats")

    st.markdown("### Medical and Lifestyle Factors")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        high_bp = yes_no_input("High blood pressure", "high_bp")
        high_cholesterol = yes_no_input("High cholesterol", "high_cholesterol")
    with m2:
        diabetes = yes_no_input("Diabetes", "diabetes")
        family_history = yes_no_input("Family history", "family_history")
    with m3:
        smoking = yes_no_input("Smoking", "smoking")
        obesity = yes_no_input("Obesity", "obesity")
    with m4:
        sedentary = yes_no_input("Sedentary lifestyle", "sedentary")
        chronic_stress = yes_no_input("Chronic stress", "chronic_stress")

    patient = pd.DataFrame(
        [
            {
                "Chest_Pain": chest_pain,
                "Shortness_of_Breath": shortness,
                "Fatigue": fatigue,
                "Palpitations": palpitations,
                "Dizziness": dizziness,
                "Swelling": swelling,
                "Pain_Arms_Jaw_Back": arms_pain,
                "Cold_Sweats_Nausea": cold_sweats,
                "High_BP": high_bp,
                "High_Cholesterol": high_cholesterol,
                "Diabetes": diabetes,
                "Smoking": smoking,
                "Obesity": obesity,
                "Sedentary_Lifestyle": sedentary,
                "Family_History": family_history,
                "Chronic_Stress": chronic_stress,
                "Gender": gender,
                "Age": age,
            }
        ]
    )
    return add_engineered_features(patient)


# =============================
# Load app data
# =============================
try:
    raw_df = load_data()
    df = clean_and_engineer(raw_df)
    trained = train_models(df)
except FileNotFoundError:
    st.markdown(
        """
        <div class="main-title">
            <h1>Heart Disease Risk Prediction</h1>
            <p>Dataset file is missing.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.error("ملف البيانات غير موجود داخل مجلد المشروع.")
    st.code(DATA_FILE)
    st.info("ضع ملف CSV بنفس الاسم بجانب app.py ثم شغل التطبيق. لا يوجد خيار رفع ملف داخل التطبيق حسب طلبك.")
    st.stop()
except Exception as error:
    st.error("صار خطأ أثناء تجهيز البيانات أو تدريب المودلات.")
    st.exception(error)
    st.stop()

models = trained["models"]
metrics_df = trained["metrics"]
feature_columns = trained["feature_columns"]
confusion_matrices = trained["confusion_matrices"]


# =============================
# Sidebar
# =============================
with st.sidebar:
    st.markdown("## ❤️ Heart Risk App")
    st.markdown(
        "<div class='small-note'>Professional ML dashboard ready for GitHub.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "القوائم",
        [
            "Overview",
            "Risk Prediction",
            "Model Comparison",
            "Data Insights",
            "About Project",
        ],
        index=0,
    )
    st.markdown("---")
    st.caption("Dataset is loaded from the project folder on GitHub.")


# =============================
# Header
# =============================
st.markdown(
    """
    <div class="main-title">
        <h1>Heart Disease Risk Prediction Dashboard</h1>
        <p>
        A clean Streamlit app for data insights, model comparison, and patient risk prediction
        using Logistic Regression and Random Forest.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================
# Pages
# =============================
if page == "Overview":
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Rows After Cleaning", f"{len(df):,}", "Duplicates removed")
    with c2:
        metric_card("Features Used", f"{len(feature_columns)}", "Original + engineered features")
    with c3:
        metric_card("PCA Components", f"{trained['pca_components']}", f"{trained['explained_variance']:.1%} variance kept")
    with c4:
        best_row = metrics_df.sort_values("Accuracy", ascending=False).iloc[0]
        metric_card("Best Model", best_row["Model"], f"Accuracy: {best_row['Accuracy']:.2%}")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Project Summary")
    st.write(
        "This project predicts heart disease risk using binary medical symptoms, lifestyle factors, age, and gender. "
        "The data is cleaned, duplicate rows are removed, engineered features are added, then two models are trained and compared."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.plotly_chart(make_risk_distribution_chart(df), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Risk Prediction":
    st.markdown('<div class="section-title">Risk Prediction</div>', unsafe_allow_html=True)

    left, right = st.columns([1.18, 0.82], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        selected_model_name = st.selectbox("Choose the model", list(models.keys()))
        patient_data = build_patient_input()
        calculate = st.button("Calculate Risk Percentage")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Prediction Result")
        if calculate:
            patient_data = patient_data[feature_columns]
            selected_model = models[selected_model_name]
            risk_probability = selected_model.predict_proba(patient_data)[0][1] * 100
            predicted_class = selected_model.predict(patient_data)[0]

            predicted_class_label = binary_label(predicted_class)
            predicted_class_css = "predicted-risk" if predicted_class_label == "Risk" else "predicted-no-risk"

            status, style_class, status_text = prediction_status(risk_probability)

            st.plotly_chart(make_gauge(risk_probability), use_container_width=True)
            st.markdown(
                f"""
                <div class="{style_class}">
                    <div class="result-title">{status}</div>
                    <p class="result-text">{status_text}</p>
                    <p class="result-text"><b>Selected Model:</b> {selected_model_name}</p>

                    <div class="predicted-class-card">
                        <div class="predicted-class-label">Predicted Class</div>
                        <div class="predicted-class-value {predicted_class_css}">{predicted_class_label}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.plotly_chart(make_gauge(0), use_container_width=True)
            st.markdown(
                "<div class='small-note'>Choose a model, answer the questions, then click the button to show the risk percentage.</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Model Comparison":
    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 0.8], gap="large")
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(make_model_comparison_chart(metrics_df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Scores Table")
        score_table = metrics_df.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
            score_table[col] = score_table[col].map(lambda x: f"{x:.2%}")
        st.dataframe(score_table, use_container_width=True, hide_index=True)
        st.markdown(
            "<div class='small-note'>The chart compares the two trained models using the same train/test split, scaling, and PCA process.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    cm1, cm2 = st.columns(2)
    with cm1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(
            make_confusion_matrix_chart(confusion_matrices["Logistic Regression"], "Confusion Matrix: Logistic Regression"),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with cm2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(
            make_confusion_matrix_chart(confusion_matrices["Random Forest"], "Confusion Matrix: Random Forest"),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Data Insights":
    st.markdown('<div class="section-title">Data Insights</div>', unsafe_allow_html=True)
    st.markdown(
        "<div class='small-note'>Gender is not used in the dashboard charts, so changing gender in prediction will not affect these visuals.</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(make_age_chart(df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(make_engineered_features_chart(df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.plotly_chart(make_correlation_chart(df), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "About Project":
    st.markdown('<div class="section-title">About Project</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### What the App Does")
    st.write(
        "The app reads the heart disease dataset directly from the project folder, cleans it, adds engineered features, "
        "trains Logistic Regression and Random Forest models, compares their metrics, and gives a risk percentage from user answers."
    )

    st.markdown("### Main Steps")
    st.markdown(
        """
        1. Load `heart_disease_risk_dataset_earlymed.csv` from the same folder as `app.py`.
        2. Convert values to integers and remove duplicate rows.
        3. Add `symptom`, `risk_symptom`, and `lifestyle` features.
        4. Split the data using stratified train/test split.
        5. Apply StandardScaler and PCA.
        6. Train Logistic Regression and Random Forest.
        7. Compare metrics and show a prediction percentage.
        """
    )

    st.info("Educational project only. This app is not a medical diagnosis tool.")
    st.markdown("</div>", unsafe_allow_html=True)
