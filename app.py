import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# =============================
# Page settings
# =============================
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# Custom CSS
# =============================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.12), transparent 35%),
            radial-gradient(circle at top right, rgba(34, 197, 94, 0.09), transparent 30%),
            linear-gradient(135deg, #07111f 0%, #0f172a 55%, #111827 100%);
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07111f 0%, #0f172a 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    .main-title {
        font-size: 44px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 4px;
        letter-spacing: -0.03em;
    }

    .sub-title {
        font-size: 18px;
        color: #cbd5e1;
        margin-bottom: 28px;
        line-height: 1.7;
    }

    .section-title {
        font-size: 26px;
        font-weight: 800;
        color: #f8fafc;
        margin: 18px 0 10px 0;
    }

    .soft-card {
        background: rgba(15, 23, 42, 0.76);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 18px 40px rgba(0,0,0,0.22);
        backdrop-filter: blur(10px);
        margin-bottom: 18px;
    }

    .metric-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.92), rgba(15, 23, 42, 0.92));
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 20px;
        padding: 20px;
        min-height: 128px;
        box-shadow: 0 16px 35px rgba(0,0,0,0.20);
    }

    .metric-label {
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 34px;
        font-weight: 800;
        line-height: 1;
    }

    .metric-note {
        color: #cbd5e1;
        font-size: 13px;
        margin-top: 8px;
    }

    .result-high {
        background: linear-gradient(145deg, rgba(127, 29, 29, 0.72), rgba(15, 23, 42, 0.86));
        border: 1px solid rgba(248, 113, 113, 0.35);
        border-radius: 22px;
        padding: 24px;
    }

    .result-low {
        background: linear-gradient(145deg, rgba(20, 83, 45, 0.70), rgba(15, 23, 42, 0.86));
        border: 1px solid rgba(74, 222, 128, 0.35);
        border-radius: 22px;
        padding: 24px;
    }

    .result-medium {
        background: linear-gradient(145deg, rgba(120, 53, 15, 0.70), rgba(15, 23, 42, 0.86));
        border: 1px solid rgba(251, 191, 36, 0.35);
        border-radius: 22px;
        padding: 24px;
    }

    .result-title {
        font-size: 22px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 6px;
    }

    .result-percent {
        font-size: 52px;
        font-weight: 900;
        color: #f8fafc;
        line-height: 1;
    }

    .small-muted {
        color: #94a3b8;
        font-size: 14px;
        line-height: 1.7;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
    }

    .stButton > button {
        background: linear-gradient(90deg, #0ea5e9, #22c55e);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.7rem 1.1rem;
        font-weight: 800;
        transition: 0.25s ease;
        box-shadow: 0 12px 30px rgba(14, 165, 233, 0.24);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        border: none;
        color: white;
        box-shadow: 0 16px 35px rgba(34, 197, 94, 0.25);
    }

    hr {
        border: none;
        border-top: 1px solid rgba(148, 163, 184, 0.18);
        margin: 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =============================
# Constants
# =============================
DATA_FILE = "heart_disease_risk_dataset_earlymed.csv"

SYMPTOM_COLUMNS = [
    "Chest_Pain",
    "Shortness_of_Breath",
    "Fatigue",
    "Palpitations",
    "Dizziness",
    "Swelling",
    "Pain_Arms_Jaw_Back",
    "Cold_Sweats_Nausea",
]

RISK_SYMPTOM_COLUMNS = [
    "Chest_Pain",
    "Shortness_of_Breath",
    "Pain_Arms_Jaw_Back",
    "Cold_Sweats_Nausea",
    "Dizziness",
]

LIFESTYLE_COLUMNS = [
    "Smoking",
    "Obesity",
    "Sedentary_Lifestyle",
    "Chronic_Stress",
]

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

FEATURE_LABELS = {
    "Chest_Pain": "Chest pain / ألم في الصدر",
    "Shortness_of_Breath": "Shortness of breath / ضيق تنفس",
    "Fatigue": "Fatigue / تعب",
    "Palpitations": "Palpitations / خفقان",
    "Dizziness": "Dizziness / دوخة",
    "Swelling": "Swelling / تورم",
    "Pain_Arms_Jaw_Back": "Pain in arms, jaw, or back / ألم في الذراع أو الفك أو الظهر",
    "Cold_Sweats_Nausea": "Cold sweats or nausea / تعرق بارد أو غثيان",
    "High_BP": "High blood pressure / ضغط مرتفع",
    "High_Cholesterol": "High cholesterol / كوليسترول مرتفع",
    "Diabetes": "Diabetes / سكري",
    "Smoking": "Smoking / تدخين",
    "Obesity": "Obesity / سمنة",
    "Sedentary_Lifestyle": "Sedentary lifestyle / قلة حركة",
    "Family_History": "Family history / تاريخ عائلي",
    "Chronic_Stress": "Chronic stress / توتر مزمن",
}


# =============================
# Helper functions
# =============================
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_FILE)
    df = df.astype(int)
    df = df.drop_duplicates()

    df["symptom"] = df[SYMPTOM_COLUMNS].sum(axis=1)
    df["risk_symptom"] = df[RISK_SYMPTOM_COLUMNS].sum(axis=1)
    df["lifestyle"] = df[LIFESTYLE_COLUMNS].sum(axis=1)

    return df


@st.cache_resource(show_spinner=False)
def train_models(df):
    X = df.drop("Heart_Risk", axis=1)
    y = df["Heart_Risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
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
            n_jobs=1
        )
    }

    metrics = []
    confusion_matrices = {}

    for model_name, model in models.items():
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)

        metrics.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
        })

        confusion_matrices[model_name] = confusion_matrix(y_test, y_pred)

    metrics_df = pd.DataFrame(metrics)

    return {
        "models": models,
        "metrics": metrics_df,
        "confusion_matrices": confusion_matrices,
        "scaler": scaler,
        "pca": pca,
        "feature_columns": X.columns.tolist(),
        "pca_components": X_train_pca.shape[1],
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
    }


def yes_no_input(label, key):
    return 1 if st.selectbox(label, ["No", "Yes"], key=key) == "Yes" else 0


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def make_metric_chart(metrics_df):
    long_df = metrics_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
        var_name="Metric",
        value_name="Score"
    )

    fig = px.bar(
        long_df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text=long_df["Score"].map(lambda x: f"{x:.3f}"),
        title="Model Performance Comparison",
        color_discrete_map={
            "Logistic Regression": "#38bdf8",
            "Random Forest": "#22c55e",
        }
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        font=dict(color="#e5e7eb"),
        title_font=dict(size=22, color="#f8fafc"),
        yaxis=dict(range=[0, 1.02], gridcolor="rgba(148,163,184,0.18)"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.18)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=20, r=20, t=70, b=20),
        height=460,
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    return fig


def make_confusion_matrix_chart(cm, model_name):
    labels = ["No Risk", "Risk"]

    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels,
        y=labels,
        color_continuous_scale=["#0f172a", "#38bdf8"],
        title=f"Confusion Matrix - {model_name}",
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        font=dict(color="#e5e7eb"),
        title_font=dict(size=20, color="#f8fafc"),
        margin=dict(l=20, r=20, t=70, b=20),
        height=430,
    )
    return fig


def make_distribution_chart(df):
    risk_counts = df["Heart_Risk"].value_counts().reset_index()
    risk_counts.columns = ["Heart_Risk", "Count"]
    risk_counts["Status"] = risk_counts["Heart_Risk"].map({0: "No Risk", 1: "Risk"})

    fig = px.bar(
        risk_counts,
        x="Status",
        y="Count",
        text="Count",
        color="Status",
        title="Heart Risk Distribution",
        color_discrete_map={
            "No Risk": "#38bdf8",
            "Risk": "#f97316",
        }
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        font=dict(color="#e5e7eb"),
        title_font=dict(size=22, color="#f8fafc"),
        showlegend=False,
        yaxis=dict(gridcolor="rgba(148,163,184,0.18)"),
        margin=dict(l=20, r=20, t=70, b=20),
        height=420,
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    return fig


def make_age_chart(df):
    chart_df = df.copy()
    chart_df["Heart Risk"] = chart_df["Heart_Risk"].map({0: "No Risk", 1: "Risk"})

    fig = px.histogram(
        chart_df,
        x="Age",
        color="Heart Risk",
        nbins=32,
        barmode="overlay",
        opacity=0.72,
        title="Age Distribution by Heart Risk",
        color_discrete_map={
            "No Risk": "#38bdf8",
            "Risk": "#f97316",
        }
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        font=dict(color="#e5e7eb"),
        title_font=dict(size=22, color="#f8fafc"),
        legend_title_text="Heart Risk",
        yaxis=dict(gridcolor="rgba(148,163,184,0.18)"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.18)"),
        margin=dict(l=20, r=20, t=70, b=20),
        height=430,
    )
    return fig


def make_correlation_chart(df):
    corr_cols = ["Age", "symptom", "risk_symptom", "lifestyle", "Heart_Risk"]
    corr = df[corr_cols].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=["#0f172a", "#38bdf8", "#22c55e"],
        title="Correlation Heatmap",
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        font=dict(color="#e5e7eb"),
        title_font=dict(size=22, color="#f8fafc"),
        margin=dict(l=20, r=20, t=70, b=20),
        height=470,
    )
    return fig


def prediction_status(probability):
    if probability >= 65:
        return "High Risk", "result-high", "The selected model predicts a high heart disease risk."
    if probability >= 35:
        return "Medium Risk", "result-medium", "The selected model predicts a medium heart disease risk."
    return "Low Risk", "result-low", "The selected model predicts a low heart disease risk."


# =============================
# Load and train
# =============================
try:
    df = load_data()
except FileNotFoundError:
    st.markdown('<div class="main-title">Heart Disease Risk Prediction</div>', unsafe_allow_html=True)
    st.error(
        "ملف البيانات غير موجود. ضع ملف البيانات في نفس مجلد التطبيق بنفس هذا الاسم:"
    )
    st.code(DATA_FILE)
    st.info(
        "لا يوجد خيار رفع ملف داخل التطبيق حسب طلبك. فقط حط ملف CSV بجانب app.py ثم شغل التطبيق."
    )
    st.stop()
except Exception as error:
    st.error("صار خطأ أثناء قراءة البيانات.")
    st.exception(error)
    st.stop()

trained = train_models(df)
models = trained["models"]
metrics_df = trained["metrics"]
feature_columns = trained["feature_columns"]


# =============================
# Sidebar
# =============================
with st.sidebar:
    st.markdown("## ❤️ Heart Risk App")
    st.markdown(
        "<span class='small-muted'>Professional Streamlit dashboard for heart disease risk prediction.</span>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    selected_page = st.radio(
        "القوائم",
        [
            "Overview",
            "Risk Prediction",
            "Model Comparison",
            "Data Insights",
            "About Project",
        ],
        index=0
    )

    st.markdown("---")
    selected_model_name = st.selectbox(
        "اختر المودل للتنبؤ",
        list(models.keys()),
        index=0
    )

    st.caption("اختيار الجنس موجود في صفحة التنبؤ فقط، ولا يتم استخدامه في الرسومات.")


# =============================
# Header
# =============================
st.markdown('<div class="main-title">Heart Disease Risk Prediction</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="sub-title">
    Dashboard احترافي للتنبؤ بخطر الإصابة بأمراض القلب باستخدام Logistic Regression و Random Forest.
    التطبيق مقسم، واضح، بدون خيار رفع ملف، وجاهز للرفع على GitHub.
    </div>
    """,
    unsafe_allow_html=True
)


# =============================
# Pages
# =============================
if selected_page == "Overview":
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Rows", f"{len(df):,}", "After removing duplicates")
    with c2:
        metric_card("Features", f"{len(feature_columns)}", "Including engineered features")
    with c3:
        metric_card("PCA Components", f"{trained['pca_components']}", f"{trained['explained_variance']:.1%} variance kept")
    with c4:
        best_row = metrics_df.sort_values("Accuracy", ascending=False).iloc[0]
        metric_card("Best Model", best_row["Model"], f"Accuracy: {best_row['Accuracy']:.2%}")

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.plotly_chart(make_distribution_chart(df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown("### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


elif selected_page == "Risk Prediction":
    st.markdown('<div class="section-title">Risk Prediction</div>', unsafe_allow_html=True)

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown("### Answer the questions")

        with st.form("prediction_form"):
            st.markdown("#### Personal Information")
            age = st.slider("Age / العمر", min_value=18, max_value=100, value=50, step=1)
            gender_text = st.selectbox("Gender / الجنس", ["Female", "Male"])
            gender = 1 if gender_text == "Male" else 0

            st.markdown("#### Symptoms")
            col1, col2 = st.columns(2)
            with col1:
                chest_pain = yes_no_input(FEATURE_LABELS["Chest_Pain"], "chest_pain")
                fatigue = yes_no_input(FEATURE_LABELS["Fatigue"], "fatigue")
                dizziness = yes_no_input(FEATURE_LABELS["Dizziness"], "dizziness")
                pain_arms_jaw_back = yes_no_input(FEATURE_LABELS["Pain_Arms_Jaw_Back"], "pain_arms")
            with col2:
                shortness_of_breath = yes_no_input(FEATURE_LABELS["Shortness_of_Breath"], "breath")
                palpitations = yes_no_input(FEATURE_LABELS["Palpitations"], "palpitations")
                swelling = yes_no_input(FEATURE_LABELS["Swelling"], "swelling")
                cold_sweats_nausea = yes_no_input(FEATURE_LABELS["Cold_Sweats_Nausea"], "cold_sweats")

            st.markdown("#### Health and Lifestyle Factors")
            col3, col4 = st.columns(2)
            with col3:
                high_bp = yes_no_input(FEATURE_LABELS["High_BP"], "high_bp")
                diabetes = yes_no_input(FEATURE_LABELS["Diabetes"], "diabetes")
                obesity = yes_no_input(FEATURE_LABELS["Obesity"], "obesity")
                family_history = yes_no_input(FEATURE_LABELS["Family_History"], "family_history")
            with col4:
                high_cholesterol = yes_no_input(FEATURE_LABELS["High_Cholesterol"], "cholesterol")
                smoking = yes_no_input(FEATURE_LABELS["Smoking"], "smoking")
                sedentary_lifestyle = yes_no_input(FEATURE_LABELS["Sedentary_Lifestyle"], "sedentary")
                chronic_stress = yes_no_input(FEATURE_LABELS["Chronic_Stress"], "stress")

            submitted = st.form_submit_button("Predict Risk")

        st.markdown('</div>', unsafe_allow_html=True)

    input_data = {
        "Chest_Pain": chest_pain,
        "Shortness_of_Breath": shortness_of_breath,
        "Fatigue": fatigue,
        "Palpitations": palpitations,
        "Dizziness": dizziness,
        "Swelling": swelling,
        "Pain_Arms_Jaw_Back": pain_arms_jaw_back,
        "Cold_Sweats_Nausea": cold_sweats_nausea,
        "High_BP": high_bp,
        "High_Cholesterol": high_cholesterol,
        "Diabetes": diabetes,
        "Smoking": smoking,
        "Obesity": obesity,
        "Sedentary_Lifestyle": sedentary_lifestyle,
        "Family_History": family_history,
        "Chronic_Stress": chronic_stress,
        "Gender": gender,
        "Age": age,
    }

    input_data["symptom"] = sum(input_data[col] for col in SYMPTOM_COLUMNS)
    input_data["risk_symptom"] = sum(input_data[col] for col in RISK_SYMPTOM_COLUMNS)
    input_data["lifestyle"] = sum(input_data[col] for col in LIFESTYLE_COLUMNS)

    input_df = pd.DataFrame([input_data])[feature_columns]

    with right:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown("### Prediction Result")

        if submitted:
            scaled_input = trained["scaler"].transform(input_df)
            pca_input = trained["pca"].transform(scaled_input)

            selected_model = models[selected_model_name]
            probability = selected_model.predict_proba(pca_input)[0][1] * 100
            status, css_class, message = prediction_status(probability)

            st.markdown(
                f"""
                <div class="{css_class}">
                    <div class="result-title">{status}</div>
                    <div class="result-percent">{probability:.1f}%</div>
                    <div class="small-muted">{message}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(float(probability / 100))

            st.markdown("#### Model used")
            st.info(selected_model_name)

            all_model_probs = []
            for model_name, model in models.items():
                all_model_probs.append({
                    "Model": model_name,
                    "Risk Probability": model.predict_proba(pca_input)[0][1] * 100
                })

            prob_df = pd.DataFrame(all_model_probs)
            fig = px.bar(
                prob_df,
                x="Risk Probability",
                y="Model",
                orientation="h",
                text=prob_df["Risk Probability"].map(lambda x: f"{x:.1f}%"),
                title="Prediction Probability by Model",
                color="Model",
                color_discrete_map={
                    "Logistic Regression": "#38bdf8",
                    "Random Forest": "#22c55e",
                }
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.35)",
                font=dict(color="#e5e7eb"),
                title_font=dict(size=18, color="#f8fafc"),
                xaxis=dict(range=[0, 100], gridcolor="rgba(148,163,184,0.18)"),
                yaxis=dict(gridcolor="rgba(148,163,184,0.18)"),
                showlegend=False,
                margin=dict(l=20, r=20, t=60, b=20),
                height=300,
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

            st.caption("This app is for educational purposes and is not a medical diagnosis.")
        else:
            st.info("اختر الإجابات من النموذج ثم اضغط Predict Risk عشان تظهر نسبة الإصابة.")

        st.markdown('</div>', unsafe_allow_html=True)


elif selected_page == "Model Comparison":
    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.plotly_chart(make_metric_chart(metrics_df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown("### Metrics Table")
    display_metrics = metrics_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        display_metrics[col] = display_metrics[col].map(lambda x: f"{x:.2%}")
    st.dataframe(display_metrics, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.plotly_chart(
            make_confusion_matrix_chart(
                trained["confusion_matrices"]["Logistic Regression"],
                "Logistic Regression"
            ),
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.plotly_chart(
            make_confusion_matrix_chart(
                trained["confusion_matrices"]["Random Forest"],
                "Random Forest"
            ),
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)


elif selected_page == "Data Insights":
    st.markdown('<div class="section-title">Data Insights</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.plotly_chart(make_age_chart(df), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.plotly_chart(make_correlation_chart(df), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown("### Project Insights")
    st.markdown(
        """
        - The number of symptoms has a clear relationship with heart risk.
        - Age is an important factor because higher-risk cases appear more often in older ages.
        - Lifestyle features such as smoking, obesity, sedentary lifestyle, and chronic stress help explain risk.
        - Logistic Regression and Random Forest both performed very well, with Logistic Regression slightly higher in this dataset.
        - The train/test results are close, which means there is no clear overfitting.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)


elif selected_page == "About Project":
    st.markdown('<div class="section-title">About Project</div>', unsafe_allow_html=True)

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown(
        """
        ### Project Summary

        This Streamlit app is based on a machine learning project for early heart disease risk prediction.

        **Main steps used in the project:**
        1. Load and clean the dataset.
        2. Remove duplicate rows.
        3. Add engineered features:
           - `symptom`
           - `risk_symptom`
           - `lifestyle`
        4. Scale the features using `StandardScaler`.
        5. Apply PCA and keep 95% of the variance.
        6. Train two models:
           - Logistic Regression
           - Random Forest
        7. Compare both models using Accuracy, Precision, Recall, and F1 Score.

        ### Important Note
        This app is only for educational use. It should not be used as a real medical diagnosis.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown("### GitHub Run Command")
    st.code("streamlit run app.py", language="bash")
    st.markdown('</div>', unsafe_allow_html=True)
