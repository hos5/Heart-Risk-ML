import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="🫀",
    layout="wide"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #F7FAFC 0%, #EEF5F9 100%);
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #183B56;
        margin-bottom: 0px;
    }

    .sub-title {
        font-size: 18px;
        color: #627D98;
        margin-top: 5px;
        margin-bottom: 22px;
    }

    .section-title {
        font-size: 27px;
        font-weight: 750;
        color: #183B56;
        margin-top: 8px;
        margin-bottom: 8px;
    }

    .card {
        background: #FFFFFF;
        padding: 22px;
        border-radius: 18px;
        border: 1px solid #D9E2EC;
        box-shadow: 0px 8px 22px rgba(16, 42, 67, 0.07);
        margin-bottom: 18px;
    }

    .metric-card {
        background: #FFFFFF;
        padding: 20px;
        border-radius: 18px;
        border: 1px solid #D9E2EC;
        box-shadow: 0px 8px 22px rgba(16, 42, 67, 0.06);
        text-align: center;
        min-height: 118px;
    }

    .metric-number {
        font-size: 32px;
        font-weight: 800;
        color: #0F766E;
        margin-bottom: 0px;
    }

    .metric-label {
        font-size: 15px;
        color: #627D98;
    }

    .risk-low {
        background: #E8F7F0;
        color: #0E7A4F;
        border: 1px solid #B7E4CC;
        border-radius: 18px;
        padding: 22px;
        text-align: center;
    }

    .risk-medium {
        background: #FFF7E6;
        color: #A15C00;
        border: 1px solid #F8D49D;
        border-radius: 18px;
        padding: 22px;
        text-align: center;
    }

    .risk-high {
        background: #FDECEC;
        color: #B42318;
        border: 1px solid #F5B5B0;
        border-radius: 18px;
        padding: 22px;
        text-align: center;
    }

    div[data-testid="stMetricValue"] {
        color: #0F766E;
    }

    .small-note {
        color: #627D98;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


# =========================
# Constants
# =========================
DATA_PATH = Path("heart_disease_risk_dataset_earlymed.csv")

BASE_COLUMNS = [
    "Chest_Pain", "Shortness_of_Breath", "Fatigue", "Palpitations",
    "Dizziness", "Swelling", "Pain_Arms_Jaw_Back", "Cold_Sweats_Nausea",
    "High_BP", "High_Cholesterol", "Diabetes", "Smoking", "Obesity",
    "Sedentary_Lifestyle", "Family_History", "Chronic_Stress", "Gender",
    "Age", "Heart_Risk"
]

SYMPTOM_COLUMNS = [
    "Chest_Pain", "Shortness_of_Breath", "Fatigue", "Palpitations",
    "Dizziness", "Swelling", "Pain_Arms_Jaw_Back", "Cold_Sweats_Nausea"
]

RISK_SYMPTOM_COLUMNS = [
    "Chest_Pain", "Shortness_of_Breath", "Pain_Arms_Jaw_Back",
    "Cold_Sweats_Nausea", "Dizziness"
]

LIFESTYLE_COLUMNS = [
    "Smoking", "Obesity", "Sedentary_Lifestyle", "Chronic_Stress"
]

MODEL_COLORS = {
    "Logistic Regression": "#0F766E",
    "Random Forest": "#4C78A8"
}

CLASS_COLORS = {
    "No Risk": "#4C78A8",
    "Risk": "#E45756"
}


# =========================
# Helper Functions
# =========================
def page_header(title: str, subtitle: str):
    st.markdown(f"<div class='main-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sub-title'>{subtitle}</div>", unsafe_allow_html=True)


def section_title(title: str):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)


def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def yes_no(label: str, key: str, help_text: str | None = None) -> int:
    value = st.selectbox(label, ["No", "Yes"], key=key, help=help_text)
    return 1 if value == "Yes" else 0


@st.cache_data
def load_local_data() -> pd.DataFrame | None:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


def validate_columns(df: pd.DataFrame) -> list[str]:
    missing = [col for col in BASE_COLUMNS if col not in df.columns]
    return missing


@st.cache_data
def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert values to numeric then integer because the dataset is 0/1 with Age.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    df = df.drop_duplicates()

    for col in df.columns:
        df[col] = df[col].astype(int)

    # Feature Engineering
    df["symptom"] = df[SYMPTOM_COLUMNS].sum(axis=1)
    df["risk_symptom"] = df[RISK_SYMPTOM_COLUMNS].sum(axis=1)
    df["lifestyle"] = df[LIFESTYLE_COLUMNS].sum(axis=1)

    return df


@st.cache_resource
def train_models(df: pd.DataFrame):
    X = df.drop("Heart_Risk", axis=1)
    y = df["Heart_Risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
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

    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train_pca, y_train)

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_pca, y_train)

    models = {
        "Logistic Regression": logistic_model,
        "Random Forest": rf_model
    }

    predictions = {
        "Logistic Regression": logistic_model.predict(X_test_pca),
        "Random Forest": rf_model.predict(X_test_pca)
    }

    metrics = []
    for model_name, y_pred in predictions.items():
        metrics.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred)
        })

    metrics_df = pd.DataFrame(metrics)

    train_test_scores = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Train Score": [
            recall_score(y_train, logistic_model.predict(X_train_pca)),
            accuracy_score(y_train, rf_model.predict(X_train_pca))
        ],
        "Test Score": [
            recall_score(y_test, logistic_model.predict(X_test_pca)),
            accuracy_score(y_test, rf_model.predict(X_test_pca))
        ],
        "Score Type": ["Recall", "Accuracy"]
    })

    return {
        "X_columns": X.columns.tolist(),
        "scaler": scaler,
        "pca": pca,
        "models": models,
        "metrics_df": metrics_df,
        "train_test_scores": train_test_scores,
        "y_test": y_test,
        "predictions": predictions,
        "pca_total_variance": pca.explained_variance_ratio_.sum(),
        "pca_components": pca.n_components_,
        "original_features": X_train_scaled.shape[1]
    }


def create_user_input() -> pd.DataFrame:
    section_title("Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", min_value=18, max_value=99, value=45, step=1)

    with col2:
        gender = st.selectbox(
            "Gender value",
            options=[0, 1],
            format_func=lambda x: "0" if x == 0 else "1",
            help="The dataset uses binary values for Gender, so choose 0 or 1."
        )

    st.divider()

    section_title("Symptoms")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        chest_pain = yes_no("Chest Pain", "chest_pain")
        shortness = yes_no("Shortness of Breath", "shortness")
    with c2:
        fatigue = yes_no("Fatigue", "fatigue")
        palpitations = yes_no("Palpitations", "palpitations")
    with c3:
        dizziness = yes_no("Dizziness", "dizziness")
        swelling = yes_no("Swelling", "swelling")
    with c4:
        arms_jaw_back = yes_no("Pain in Arms/Jaw/Back", "arms_jaw_back")
        cold_sweats = yes_no("Cold Sweats/Nausea", "cold_sweats")

    st.divider()

    section_title("Health & Lifestyle Factors")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        high_bp = yes_no("High Blood Pressure", "high_bp")
        high_cholesterol = yes_no("High Cholesterol", "high_cholesterol")
    with c2:
        diabetes = yes_no("Diabetes", "diabetes")
        smoking = yes_no("Smoking", "smoking")
    with c3:
        obesity = yes_no("Obesity", "obesity")
        sedentary = yes_no("Sedentary Lifestyle", "sedentary")
    with c4:
        family_history = yes_no("Family History", "family_history")
        chronic_stress = yes_no("Chronic Stress", "chronic_stress")

    input_data = {
        "Chest_Pain": chest_pain,
        "Shortness_of_Breath": shortness,
        "Fatigue": fatigue,
        "Palpitations": palpitations,
        "Dizziness": dizziness,
        "Swelling": swelling,
        "Pain_Arms_Jaw_Back": arms_jaw_back,
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
        "Age": age
    }

    input_df = pd.DataFrame([input_data])

    input_df["symptom"] = input_df[SYMPTOM_COLUMNS].sum(axis=1)
    input_df["risk_symptom"] = input_df[RISK_SYMPTOM_COLUMNS].sum(axis=1)
    input_df["lifestyle"] = input_df[LIFESTYLE_COLUMNS].sum(axis=1)

    return input_df


def risk_box(probability: float, prediction: int):
    percent = probability * 100

    if percent < 35:
        css_class = "risk-low"
        level = "Low Risk"
        message = "The model predicts a lower probability of heart disease risk."
    elif percent < 65:
        css_class = "risk-medium"
        level = "Moderate Risk"
        message = "The model predicts a medium probability of heart disease risk."
    else:
        css_class = "risk-high"
        level = "High Risk"
        message = "The model predicts a higher probability of heart disease risk."

    st.markdown(
        f"""
        <div class="{css_class}">
            <h2>{percent:.2f}%</h2>
            <h3>{level}</h3>
            <p>{message}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<p class='small-note'>This result is for educational machine learning purposes only and is not a medical diagnosis.</p>",
        unsafe_allow_html=True
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percent,
        number={"suffix": "%"},
        title={"text": "Predicted Heart Risk Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#0F766E"},
            "steps": [
                {"range": [0, 35], "color": "#DFF5E7"},
                {"range": [35, 65], "color": "#FFF2CC"},
                {"range": [65, 100], "color": "#FAD2D2"}
            ],
            "threshold": {
                "line": {"color": "#183B56", "width": 4},
                "thickness": 0.75,
                "value": percent
            }
        }
    ))

    fig.update_layout(
        height=330,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_model_comparison(metrics_df: pd.DataFrame):
    long_df = metrics_df.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score"
    )

    y_min = max(0, long_df["Score"].min() - 0.01)

    fig = px.bar(
        long_df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text="Score",
        color_discrete_map=MODEL_COLORS,
        title="Model Performance Comparison"
    )

    fig.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_range=[y_min, 1.005],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title_text="Model",
        title_font_size=22,
        font=dict(size=13)
    )

    return fig


def plot_confusion_matrix(y_test, y_pred, model_name: str):
    cm = confusion_matrix(y_test, y_pred)

    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["No Risk", "Risk"],
        y=["No Risk", "Risk"],
        color_continuous_scale="Blues",
        title=f"Confusion Matrix - {model_name}"
    )

    fig.update_layout(
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font_size=20
    )

    return fig


# =========================
# Sidebar
# =========================
st.sidebar.title("🫀 Heart Risk App")
st.sidebar.caption("Machine Learning Dashboard")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"],
    help="If the CSV is already in the same folder as app.py, you do not need to upload it."
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Data Insights",
        "Model Performance",
        "Risk Prediction",
        "Project Notes"
    ]
)


# =========================
# Load Data
# =========================
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
else:
    raw_df = load_local_data()

if raw_df is None:
    page_header(
        "Heart Disease Risk Prediction",
        "Please upload the CSV file or place heart_disease_risk_dataset_earlymed.csv in the same folder as app.py."
    )
    st.error("Dataset was not found.")
    st.info("For GitHub, add `heart_disease_risk_dataset_earlymed.csv` in the same repository folder with `app.py`.")
    st.stop()

missing_columns = validate_columns(raw_df)
if missing_columns:
    st.error("The dataset is missing required columns:")
    st.write(missing_columns)
    st.stop()

df = clean_and_engineer(raw_df)
training = train_models(df)

df["Heart_Risk_Label"] = df["Heart_Risk"].map({0: "No Risk", 1: "Risk"})


# =========================
# Pages
# =========================
if page == "Overview":
    page_header(
        "Heart Disease Risk Prediction",
        "A professional Streamlit dashboard for EDA, model comparison, and user risk prediction."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Rows After Cleaning", f"{df.shape[0]:,}")
    with c2:
        metric_card("Features After Engineering", f"{df.shape[1] - 1}")
    with c3:
        metric_card("PCA Components", f"{training['pca_components']}")
    with c4:
        best_model = training["metrics_df"].sort_values("Accuracy", ascending=False).iloc[0]["Model"]
        metric_card("Best Model", best_model)

    st.markdown("""
    <div class="card">
        <h3>Project Summary</h3>
        <p>
        This project predicts whether a person has heart disease risk or not.
        The target column is <b>Heart_Risk</b>, where 0 means <b>No Risk</b> and 1 means <b>Risk</b>.
        The project includes data cleaning, feature engineering, scaling, PCA, model training,
        model evaluation, and a prediction page where the user can choose the model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(df.head(10), use_container_width=True)

elif page == "Data Insights":
    page_header(
        "Data Insights",
        "Explore the main patterns in the heart disease risk dataset."
    )

    col1, col2 = st.columns(2)

    with col1:
        target_counts = df["Heart_Risk_Label"].value_counts().reset_index()
        target_counts.columns = ["Heart Risk", "Count"]

        fig = px.bar(
            target_counts,
            x="Heart Risk",
            y="Count",
            color="Heart Risk",
            text="Count",
            color_discrete_map=CLASS_COLORS,
            title="Heart Risk Distribution"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info("The target classes are close in count, so the model is not strongly biased toward one class.")

    with col2:
        fig = px.histogram(
            df,
            x="Age",
            color="Heart_Risk_Label",
            nbins=35,
            marginal="box",
            barmode="overlay",
            opacity=0.70,
            color_discrete_map=CLASS_COLORS,
            title="Age Distribution by Heart Risk"
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend_title_text="Heart Risk"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info("Age shows a clear relationship with heart risk. Older people are more likely to be in the Risk class.")

    corr_cols = ["Age", "symptom", "risk_symptom", "lifestyle", "Heart_Risk"]
    corr = df[corr_cols].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Heatmap"
    )
    fig.update_layout(
        height=620,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success("Symptoms and risk symptoms have the strongest relationship with Heart_Risk, followed by lifestyle and age.")

elif page == "Model Performance":
    page_header(
        "Model Performance",
        "Compare Logistic Regression and Random Forest using evaluation metrics and confusion matrices."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Total Explained Variance", f"{training['pca_total_variance']:.2%}")
    with c2:
        metric_card("Before PCA", str(training["original_features"]))
    with c3:
        metric_card("After PCA", str(training["pca_components"]))

    st.markdown("### Metrics Table")
    st.dataframe(training["metrics_df"].style.format({
        "Accuracy": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1 Score": "{:.4f}"
    }), use_container_width=True)

    st.plotly_chart(plot_model_comparison(training["metrics_df"]), use_container_width=True)

    st.markdown("### Train vs Test Check")
    st.dataframe(training["train_test_scores"].style.format({
        "Train Score": "{:.4f}",
        "Test Score": "{:.4f}"
    }), use_container_width=True)

    st.info("Train and test scores are very close, which means there is no clear overfitting.")

    st.markdown("### Confusion Matrices")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plot_confusion_matrix(
                training["y_test"],
                training["predictions"]["Logistic Regression"],
                "Logistic Regression"
            ),
            use_container_width=True
        )
    with c2:
        st.plotly_chart(
            plot_confusion_matrix(
                training["y_test"],
                training["predictions"]["Random Forest"],
                "Random Forest"
            ),
            use_container_width=True
        )

elif page == "Risk Prediction":
    page_header(
        "Risk Prediction",
        "Choose a model and enter patient information to get the predicted heart risk probability."
    )

    st.warning("This app is for educational machine learning purposes only. It is not a medical diagnosis.")

    selected_model_name = st.selectbox(
        "Choose Model",
        ["Logistic Regression", "Random Forest"]
    )

    with st.form("prediction_form"):
        user_input = create_user_input()
        submitted = st.form_submit_button("Predict Heart Risk", use_container_width=True)

    if submitted:
        # Make sure the user input has the same order as training columns.
        user_input = user_input[training["X_columns"]]

        user_scaled = training["scaler"].transform(user_input)
        user_pca = training["pca"].transform(user_scaled)

        selected_model = training["models"][selected_model_name]
        probability = selected_model.predict_proba(user_pca)[0][1]
        prediction = selected_model.predict(user_pca)[0]

        c1, c2 = st.columns([1, 1])

        with c1:
            risk_box(probability, prediction)

        with c2:
            st.markdown("### Input Summary")
            st.dataframe(user_input.T.rename(columns={0: "Value"}), use_container_width=True)

            result_df = user_input.copy()
            result_df["Selected_Model"] = selected_model_name
            result_df["Risk_Probability"] = probability
            result_df["Prediction"] = "Risk" if prediction == 1 else "No Risk"

            st.download_button(
                label="Download Prediction Result",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="heart_risk_prediction_result.csv",
                mime="text/csv",
                use_container_width=True
            )

elif page == "Project Notes":
    page_header(
        "Project Notes",
        "Main project insights and conclusion."
    )

    st.markdown("""
    <div class="card">
        <h3>Overall Insight</h3>
        <p>
        The main insight from this project is that heart disease risk is strongly related to symptoms,
        age, and lifestyle factors. People with more symptoms and more risky lifestyle factors had a higher
        chance of being classified as Risk.
        </p>
        <p>
        Both Logistic Regression and Random Forest gave high results, but Logistic Regression was slightly better.
        The difference was small, which means both models worked well.
        </p>
        <p>
        The train and test results were also close, so there was no clear overfitting. Overall, the dataset helped
        the models predict Heart_Risk very well.
        </p>
    </div>

    <div class="card">
        <h3>Conclusion</h3>
        <p>
        In this project, I used a heart disease risk dataset to predict whether a person has heart disease risk or not.
        The target column was Heart_Risk, where 0 means No Risk and 1 means Risk.
        </p>
        <p>
        First, I explored the dataset by checking the shape, data types, missing values, duplicated rows,
        and summary statistics. I also created EDA visualizations to understand the data better and see the
        relationship between the features and Heart_Risk.
        </p>
        <p>
        After that, I added three new features: symptom, risk_symptom, and lifestyle. These features helped summarize
        the number of symptoms, heart-related symptoms, and lifestyle risk factors for each person.
        </p>
        <p>
        I trained two models: Logistic Regression and Random Forest. Both models performed very well, but Logistic
        Regression was slightly better in accuracy, precision, recall, and F1 score. Overall, Logistic Regression
        was selected as the best model because it gave the highest performance and was simple and fast.
        </p>
        <p>
        With more time, I would improve the project by trying more models, tuning the hyperparameters, and using
        cross-validation to make sure the model performs well on different data splits.
        </p>
    </div>
    """, unsafe_allow_html=True)
