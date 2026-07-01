import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# =========================
# Page Settings
# =========================
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide"
)

DATA_FILE = "heart_disease_risk_dataset_earlymed.csv"

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

TARGET = "Heart_Risk"

SYMPTOM_COLS = [
    "Chest_Pain",
    "Shortness_of_Breath",
    "Fatigue",
    "Palpitations",
    "Dizziness",
    "Swelling",
    "Pain_Arms_Jaw_Back",
    "Cold_Sweats_Nausea",
]

RISK_SYMPTOM_COLS = [
    "Chest_Pain",
    "Shortness_of_Breath",
    "Pain_Arms_Jaw_Back",
    "Cold_Sweats_Nausea",
    "Dizziness",
]

LIFESTYLE_COLS = [
    "Smoking",
    "Obesity",
    "Sedentary_Lifestyle",
    "Chronic_Stress",
]

MODEL_FEATURES = BASE_FEATURES + ["symptom", "risk_symptom", "lifestyle"]


# =========================
# Design / CSS
# =========================
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #08111f 0%, #0f1f35 45%, #0b2a2d 100%);
            color: #f8fafc;
        }

        [data-testid="stSidebar"] {
            background: #0b1220;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        h1, h2, h3 {
            color: #f8fafc !important;
            letter-spacing: 0.2px;
        }

        p, label, span, div {
            color: #e5e7eb;
        }

        .main-title {
            padding: 28px 30px;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(20,184,166,0.20), rgba(96,165,250,0.16));
            border: 1px solid rgba(255,255,255,0.10);
            margin-bottom: 24px;
            box-shadow: 0px 20px 60px rgba(0,0,0,0.25);
        }

        .main-title h1 {
            font-size: 42px;
            margin-bottom: 8px;
        }

        .main-title p {
            font-size: 18px;
            color: #cbd5e1;
            margin: 0;
        }

        .section-card {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
            padding: 22px;
            margin: 12px 0px;
            box-shadow: 0px 14px 45px rgba(0,0,0,0.22);
        }

        .small-note {
            color: #94a3b8;
            font-size: 14px;
            line-height: 1.6;
        }

        .success-box {
            padding: 18px 20px;
            background: rgba(20,184,166,0.14);
            border: 1px solid rgba(20,184,166,0.35);
            border-radius: 18px;
            color: #ccfbf1;
            font-weight: 600;
        }

        .warning-box {
            padding: 18px 20px;
            background: rgba(244,63,94,0.14);
            border: 1px solid rgba(244,63,94,0.35);
            border-radius: 18px;
            color: #ffe4e6;
            font-weight: 600;
        }

        .metric-card {
            background: rgba(2, 6, 23, 0.45);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 18px;
            text-align: center;
        }

        .metric-card h4 {
            color: #94a3b8;
            font-size: 15px;
            margin-bottom: 8px;
        }

        .metric-card h2 {
            color: #ffffff;
            font-size: 30px;
            margin: 0;
        }

        div[data-testid="stMetricValue"] {
            color: #ffffff;
        }

        div[data-testid="stTabs"] button p {
            font-size: 16px;
            font-weight: 700;
            color: #e5e7eb;
        }

        .stButton > button {
            width: 100%;
            border-radius: 14px;
            background: linear-gradient(135deg, #14b8a6, #3b82f6);
            color: white;
            border: none;
            padding: 12px 18px;
            font-weight: 700;
        }

        .stButton > button:hover {
            color: white;
            border: none;
            filter: brightness(1.08);
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# Helper Functions
# =========================
def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add the same engineered features used in the notebook."""
    data = data.copy()
    data["symptom"] = data[SYMPTOM_COLS].sum(axis=1)
    data["risk_symptom"] = data[RISK_SYMPTOM_COLS].sum(axis=1)
    data["lifestyle"] = data[LIFESTYLE_COLS].sum(axis=1)
    return data


def clean_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare the dataset."""
    df = raw_df.copy()
    df.columns = df.columns.str.strip()

    required = BASE_FEATURES + [TARGET]
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).copy()

    for col in required:
        df[col] = df[col].astype(int)

    df = df.drop_duplicates()
    df = add_engineered_features(df)

    return df


@st.cache_resource(show_spinner=False)
def train_models_from_csv(csv_bytes: bytes):
    """Load data, train both models, and return all needed artifacts."""
    raw_df = pd.read_csv(io.BytesIO(csv_bytes))
    df = clean_dataset(raw_df)

    X = df[MODEL_FEATURES]
    y = df[TARGET]

    stratify_value = y if y.nunique() == 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=stratify_value
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        )
    }

    metrics_rows = []
    confusion_matrices = {}
    predictions = {}

    for model_name, model in models.items():
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)
        predictions[model_name] = y_pred

        metrics_rows.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        })

        confusion_matrices[model_name] = confusion_matrix(y_test, y_pred)

    metrics_df = pd.DataFrame(metrics_rows)

    return {
        "df": df,
        "models": models,
        "metrics_df": metrics_df,
        "confusion_matrices": confusion_matrices,
        "scaler": scaler,
        "pca": pca,
        "model_features": MODEL_FEATURES,
        "x_test_pca": X_test_pca,
        "y_test": y_test,
        "pca_components": X_train_pca.shape[1],
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
    }


def load_csv_bytes():
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="Upload heart_disease_risk_dataset_earlymed.csv"
    )

    if uploaded_file is not None:
        return uploaded_file.getvalue(), uploaded_file.name

    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as file:
            return file.read(), DATA_FILE

    return None, None


def yes_no_checkbox(label: str) -> int:
    return int(st.checkbox(label))


def make_metric_card(title: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <h4>{title}</h4>
            <h2>{value}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


def risk_label(value: int) -> str:
    return "Risk" if value == 1 else "No Risk"


def plot_confusion_matrix(cm, title):
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted No Risk", "Predicted Risk"],
            y=["Actual No Risk", "Actual Risk"],
            text=cm,
            texttemplate="%{text}",
            colorscale=[
                [0, "#0f172a"],
                [0.5, "#2563eb"],
                [1, "#14b8a6"]
            ],
            showscale=False
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        font=dict(color="#f8fafc")
    )

    return fig


def build_patient_input(age_min: int, age_max: int) -> pd.DataFrame:
    st.markdown("#### Patient Information")

    col_age, col_gender, col_model_note = st.columns([1, 1, 1.3])

    with col_age:
        age = st.slider("Age", min_value=18, max_value=100, value=int((age_min + age_max) / 2))

    with col_gender:
        gender_text = st.selectbox("Gender", ["Female", "Male"])
        gender = 0 if gender_text == "Female" else 1

    with col_model_note:
        st.markdown(
            """
            <div class="small-note">
            Gender is used only in the prediction form. It does not filter or change any EDA charts.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("#### Symptoms")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        chest_pain = yes_no_checkbox("Chest pain")
        shortness = yes_no_checkbox("Shortness of breath")
    with s2:
        fatigue = yes_no_checkbox("Fatigue")
        palpitations = yes_no_checkbox("Palpitations")
    with s3:
        dizziness = yes_no_checkbox("Dizziness")
        swelling = yes_no_checkbox("Swelling")
    with s4:
        arms_pain = yes_no_checkbox("Pain in arms / jaw / back")
        cold_sweats = yes_no_checkbox("Cold sweats / nausea")

    st.markdown("#### Medical and Lifestyle Factors")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        high_bp = yes_no_checkbox("High blood pressure")
        high_cholesterol = yes_no_checkbox("High cholesterol")
    with m2:
        diabetes = yes_no_checkbox("Diabetes")
        family_history = yes_no_checkbox("Family history")
    with m3:
        smoking = yes_no_checkbox("Smoking")
        obesity = yes_no_checkbox("Obesity")
    with m4:
        sedentary = yes_no_checkbox("Sedentary lifestyle")
        chronic_stress = yes_no_checkbox("Chronic stress")

    patient_data = pd.DataFrame([{
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
    }])

    patient_data = add_engineered_features(patient_data)
    return patient_data


# =========================
# Header
# =========================
st.markdown(
    """
    <div class="main-title">
        <h1>Heart Disease Risk Prediction Dashboard</h1>
        <p>
        A professional Streamlit app for EDA, model comparison, and patient risk prediction
        using Logistic Regression and Random Forest.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# Sidebar
# =========================
st.sidebar.title("Project Controls")
st.sidebar.markdown(
    """
    Upload your dataset or keep the CSV file in the same folder as `app.py`.

    Expected file name:
    `heart_disease_risk_dataset_earlymed.csv`
    """
)

csv_bytes, csv_name = load_csv_bytes()

if csv_bytes is None:
    st.error(
        "Dataset not found. Please upload the CSV file from the sidebar or place "
        "`heart_disease_risk_dataset_earlymed.csv` in the same folder as `app.py`."
    )

    st.info(
        "Required columns: "
        + ", ".join(BASE_FEATURES + [TARGET])
    )
    st.stop()

try:
    with st.spinner("Training models and preparing dashboard..."):
        artifacts = train_models_from_csv(csv_bytes)
except Exception as error:
    st.error("There is a problem while loading or training the dataset.")
    st.exception(error)
    st.stop()

df = artifacts["df"]
models = artifacts["models"]
metrics_df = artifacts["metrics_df"]


# =========================
# KPIs
# =========================
risk_count = int(df[TARGET].sum())
no_risk_count = int((df[TARGET] == 0).sum())
risk_rate = df[TARGET].mean() * 100

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    make_metric_card("Total Records", f"{len(df):,}")
with kpi2:
    make_metric_card("Heart Risk Cases", f"{risk_count:,}")
with kpi3:
    make_metric_card("No Risk Cases", f"{no_risk_count:,}")
with kpi4:
    make_metric_card("Risk Rate", f"{risk_rate:.1f}%")


# =========================
# Tabs
# =========================
overview_tab, eda_tab, model_tab, prediction_tab, conclusion_tab = st.tabs(
    [
        "Overview",
        "Visual Analysis",
        "Model Comparison",
        "Risk Prediction",
        "Conclusion",
    ]
)


# =========================
# Overview Tab
# =========================
with overview_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Dataset Overview")

    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.write("Dataset file:", csv_name)
        st.dataframe(df.head(10), use_container_width=True)

    with c2:
        st.markdown("#### Project Features")
        st.markdown(
            """
            - Data cleaning and duplicate removal.
            - Feature engineering for symptoms, risky symptoms, and lifestyle.
            - PCA with 95% explained variance.
            - Logistic Regression and Random Forest models.
            - Interactive patient risk prediction.
            """
        )

        st.markdown("#### PCA Summary")
        st.write(f"Number of PCA components used: **{artifacts['pca_components']}**")
        st.write(f"Explained variance: **{artifacts['explained_variance']:.2%}**")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# EDA Tab
# =========================
with eda_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Visual Analysis")

    df_plot = df.copy()
    df_plot["Risk_Label"] = df_plot[TARGET].map({0: "No Risk", 1: "Risk"})

    c1, c2 = st.columns(2)

    with c1:
        risk_counts = (
            df_plot["Risk_Label"]
            .value_counts()
            .reindex(["No Risk", "Risk"])
            .reset_index()
        )
        risk_counts.columns = ["Risk_Label", "Count"]

        fig_risk = px.bar(
            risk_counts,
            x="Risk_Label",
            y="Count",
            text="Count",
            color="Risk_Label",
            color_discrete_map={
                "No Risk": "#14b8a6",
                "Risk": "#f43f5e"
            },
            title="Heart Risk Distribution"
        )
        fig_risk.update_traces(textposition="outside")
        fig_risk.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=430,
            font=dict(color="#f8fafc")
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with c2:
        fig_age = px.histogram(
            df_plot,
            x="Age",
            color="Risk_Label",
            nbins=30,
            barmode="overlay",
            opacity=0.72,
            color_discrete_map={
                "No Risk": "#14b8a6",
                "Risk": "#f43f5e"
            },
            title="Age Distribution by Heart Risk"
        )
        fig_age.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=430,
            font=dict(color="#f8fafc")
        )
        st.plotly_chart(fig_age, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        age_bins = [18, 30, 40, 50, 60, 70, 80, 100]
        age_labels = ["18-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81+"]
        df_plot["Age_Group"] = pd.cut(df_plot["Age"], bins=age_bins, labels=age_labels, include_lowest=True)

        age_risk = (
            df_plot
            .groupby("Age_Group", observed=False)[TARGET]
            .mean()
            .reset_index()
        )
        age_risk["Risk_Percentage"] = age_risk[TARGET] * 100

        fig_age_risk = px.line(
            age_risk,
            x="Age_Group",
            y="Risk_Percentage",
            markers=True,
            title="Heart Risk Percentage by Age Group"
        )
        fig_age_risk.update_traces(line=dict(width=4), marker=dict(size=10))
        fig_age_risk.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Risk Percentage (%)",
            xaxis_title="Age Group",
            height=430,
            font=dict(color="#f8fafc")
        )
        st.plotly_chart(fig_age_risk, use_container_width=True)

    with c4:
        corr_cols = ["Age", "symptom", "risk_symptom", "lifestyle", TARGET]
        corr = df_plot[corr_cols].corr()

        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale=[
                    [0, "#0f172a"],
                    [0.5, "#2563eb"],
                    [1, "#14b8a6"]
                ],
                zmin=-1,
                zmax=1,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
                colorbar=dict(title="Correlation")
            )
        )
        fig_corr.update_layout(
            title="Correlation Heatmap",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=430,
            font=dict(color="#f8fafc")
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Model Comparison Tab
# =========================
with model_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Model Performance Comparison")

    st.dataframe(
        metrics_df.style.format({
            "Accuracy": "{:.3%}",
            "Precision": "{:.3%}",
            "Recall": "{:.3%}",
            "F1 Score": "{:.3%}",
        }),
        use_container_width=True
    )

    metrics_long = metrics_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
        var_name="Metric",
        value_name="Score"
    )
    metrics_long["Score_Percentage"] = metrics_long["Score"] * 100

    fig_metrics = px.bar(
        metrics_long,
        x="Metric",
        y="Score_Percentage",
        color="Model",
        barmode="group",
        text=metrics_long["Score_Percentage"].map(lambda x: f"{x:.2f}%"),
        color_discrete_map={
            "Logistic Regression": "#14b8a6",
            "Random Forest": "#60a5fa"
        },
        title="Logistic Regression vs Random Forest"
    )

    fig_metrics.update_traces(textposition="outside")
    fig_metrics.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Score (%)",
        xaxis_title="Metric",
        height=500,
        font=dict(color="#f8fafc"),
        legend_title_text="Model"
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

    cm1, cm2 = st.columns(2)
    with cm1:
        st.plotly_chart(
            plot_confusion_matrix(
                artifacts["confusion_matrices"]["Logistic Regression"],
                "Confusion Matrix - Logistic Regression"
            ),
            use_container_width=True
        )

    with cm2:
        st.plotly_chart(
            plot_confusion_matrix(
                artifacts["confusion_matrices"]["Random Forest"],
                "Confusion Matrix - Random Forest"
            ),
            use_container_width=True
        )

    best_model = metrics_df.sort_values("Recall", ascending=False).iloc[0]
    st.markdown(
        f"""
        <div class="success-box">
        Best model by Recall: {best_model['Model']} with {best_model['Recall']:.2%}.
        Recall is important here because missing a real heart-risk case can be dangerous.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Prediction Tab
# =========================
with prediction_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Interactive Heart Risk Prediction")

    selected_model_name = st.selectbox(
        "Choose the model you want to use",
        list(models.keys())
    )

    patient_data = build_patient_input(
        age_min=int(df["Age"].min()),
        age_max=int(df["Age"].max())
    )

    predict_button = st.button("Predict Heart Risk")

    if predict_button:
        selected_model = models[selected_model_name]

        input_x = patient_data[artifacts["model_features"]]
        input_scaled = artifacts["scaler"].transform(input_x)
        input_pca = artifacts["pca"].transform(input_scaled)

        risk_probability = selected_model.predict_proba(input_pca)[0][1]
        prediction = selected_model.predict(input_pca)[0]

        result_col, gauge_col = st.columns([1, 1])

        with result_col:
            st.metric(
                label="Heart Risk Probability",
                value=f"{risk_probability * 100:.2f}%"
            )

            st.metric(
                label="Selected Model",
                value=selected_model_name
            )

            if prediction == 1:
                st.markdown(
                    """
                    <div class="warning-box">
                    Result: High Heart Risk. This result means the model found risk patterns in the answers.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="success-box">
                    Result: Low Heart Risk. This result means the model did not find strong risk patterns in the answers.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(
                """
                <div class="small-note">
                This app is for educational machine learning use only. It is not a medical diagnosis.
                Always consult a medical professional for real health decisions.
                </div>
                """,
                unsafe_allow_html=True
            )

        with gauge_col:
            gauge_color = "#f43f5e" if risk_probability >= 0.5 else "#14b8a6"

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_probability * 100,
                    number={"suffix": "%", "font": {"size": 46}},
                    title={"text": "Risk Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": gauge_color},
                        "steps": [
                            {"range": [0, 50], "color": "rgba(20,184,166,0.25)"},
                            {"range": [50, 100], "color": "rgba(244,63,94,0.25)"}
                        ],
                        "threshold": {
                            "line": {"color": "#ffffff", "width": 4},
                            "thickness": 0.75,
                            "value": 50
                        }
                    }
                )
            )
            fig_gauge.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=420,
                font=dict(color="#f8fafc")
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Conclusion Tab
# =========================
with conclusion_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Project Conclusion")

    st.markdown(
        """
        This project predicts heart disease risk using binary medical and lifestyle features.
        The dataset was cleaned by removing duplicates, and three new features were added:
        total symptoms, risky symptoms, and lifestyle risk score.

        Logistic Regression and Random Forest were trained after scaling and PCA.
        The comparison chart shows that both models perform very well, with very close scores.
        The prediction section allows the user to choose the model, answer patient questions,
        and get the estimated heart-risk percentage.

        Gender is included only as a prediction input and does not affect the dashboard charts.
        """
    )

    st.markdown("</div>", unsafe_allow_html=True)
