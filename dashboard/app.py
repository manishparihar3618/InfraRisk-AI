import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from streamlit_option_menu import option_menu
from sklearn.metrics import confusion_matrix
from pathlib import Path

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------

st.set_page_config(
    page_title="InfraRisk AI",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------

st.markdown("""
<style>

.main{
    background:#0E1117;
}

.block-container{
    padding-top:1rem;
}

.title{
    font-size:54px;
    font-weight:700;
    color:#2563EB;
    white-space:nowrap;
    overflow:hidden;
}

.subtitle{
    font-size:18px;
    color:#BBBBBB;
}

.metric-card{
    background:#1E293B;
    padding:18px;
    border-radius:18px;
    text-align:center;
    box-shadow:0px 0px 10px rgba(0,0,0,.25);
}

.metric-value{
    font-size:34px;
    font-weight:bold;
    color:white;
}

.metric-title{
    color:#A0AEC0;
    font-size:16px;
}

.low{
    background:#16A34A;
    color:white;
    padding:12px;
    border-radius:12px;
    text-align:center;
    font-size:22px;
    font-weight:bold;
}

.medium{
    background:#F59E0B;
    color:white;
    padding:12px;
    border-radius:12px;
    text-align:center;
    font-size:22px;
    font-weight:bold;
}

.high{
    background:#DC2626;
    color:white;
    padding:12px;
    border-radius:12px;
    text-align:center;
    font-size:22px;
    font-weight:bold;
}

footer{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# Load Files
# ----------------------------------------------------------

BASE = Path(__file__).resolve().parent.parent

DATA_PATH = BASE / "data" / "florida_bridge_with_risk_final.csv"
MODEL_PATH = BASE / "models" / "xgboost_model.pkl"
ENCODER_PATH = BASE / "models" / "label_encoder.pkl"
FEATURE_PATH = BASE / "models" / "features.pkl"

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
features = joblib.load(FEATURE_PATH)

# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------

with st.sidebar:

    st.image("https://img.icons8.com/color/96/bridge.png", width=70)

    st.markdown("## InfraRisk AI")

    selected = option_menu(
        menu_title="Navigation",
        options=[
            "Home",
            "Predict Bridge Risk",
            "Dataset Analytics",
            "Feature Importance",
            "About"
        ],
        icons=[
            "house",
            "activity",
            "bar-chart",
            "graph-up",
            "info-circle"
        ],
        default_index=0
    )

# ----------------------------------------------------------
# Header
# ----------------------------------------------------------

st.markdown(
    "<div class='title'>🏗️ InfraRisk AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>AI Powered Infrastructure Risk Assessment System using XGBoost</div>",
    unsafe_allow_html=True
)

st.write("")

# ----------------------------------------------------------
# KPI Values
# ----------------------------------------------------------

total_bridges = len(df)

low_count = (df["Infrastructure_Risk"]=="Low").sum()

medium_count = (df["Infrastructure_Risk"]=="Medium").sum()

high_count = (df["Infrastructure_Risk"]=="High").sum()

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.metric("🏗️ Total Bridges", total_bridges)

with c2:
    st.metric("🟢 Low Risk", low_count)

with c3:
    st.metric("🟡 Medium Risk", medium_count)

with c4:
    st.metric("🔴 High Risk", high_count)

st.divider()

# ==========================================================
# PART 2 starts below this...
# ==========================================================
if selected == "Home":

    st.markdown("## Dashboard Overview")

    col1, col2 = st.columns([2, 1])

    with col1:

        st.info("""
InfraRisk AI is an AI-powered Infrastructure Risk Assessment System that predicts
bridge risk using Machine Learning (XGBoost). The dashboard provides analytics,
risk prediction, feature importance and infrastructure insights.
        """)

        st.write("")

        model_df = pd.DataFrame({
            "Model": [
                "Decision Tree",
                "Random Forest",
                "XGBoost"
            ],
            "Accuracy": [
                69.4,
                73.4,
                74.7
            ]
        })

        fig = px.bar(
            model_df,
            x="Model",
            y="Accuracy",
            text="Accuracy",
            color="Accuracy",
            color_continuous_scale="Blues"
        )

        fig.update_layout(
            height=400,
            template="plotly_dark",
            coloraxis_showscale=False
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.metric("Best Model", "XGBoost")

        st.metric("Accuracy", "74.7%")

        st.metric("Dataset Size", f"{len(df):,}")

        st.metric("Features", len(features))

    st.write("")

    left, right = st.columns(2)

    with left:

        risk = df["Infrastructure_Risk"].value_counts().reset_index()
        risk.columns = ["Risk", "Count"]

        fig = px.pie(
            risk,
            names="Risk",
            values="Count",
            hole=.6,
            color="Risk",
            color_discrete_map={
                "Low":"green",
                "Medium":"orange",
                "High":"red"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        risk = df["Infrastructure_Risk"].value_counts().reset_index()
        risk.columns = ["Risk", "Count"]

        fig = px.bar(
            risk,
            x="Risk",
            y="Count",
            text="Count",
            color="Risk",
            color_discrete_map={
                "Low":"green",
                "Medium":"orange",
                "High":"red"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            height=450,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset Preview")

    st.dataframe(df.head(15), use_container_width=True)

elif selected == "Predict Bridge Risk":

    st.subheader("Bridge Risk Prediction")

    st.write("Enter bridge information below to predict infrastructure risk.")

    user_input = {}

    cols = st.columns(2)

    for i, feature in enumerate(features):

        with cols[i % 2]:

            dtype = df[feature].dtype

            if str(dtype) == "object":

                options = sorted(df[feature].dropna().unique())

                user_input[feature] = st.selectbox(
                    feature.replace("_", " "),
                    options
                )

            else:

                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                default = float(df[feature].median())

                user_input[feature] = st.number_input(
                    feature.replace("_", " "),
                    min_value=min_val,
                    max_value=max_val,
                    value=default
                )

    st.write("")

    if st.button("Predict Risk", use_container_width=True):

        input_df = pd.DataFrame([user_input])

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df).max() * 100

        risk = label_encoder.inverse_transform([prediction])[0]

        st.write("")

        if risk == "Low":

            st.markdown(
                f"""
<div style="
background:#16A34A;
padding:25px;
border-radius:18px;
text-align:center;
color:white;
font-size:28px;
font-weight:bold;">
🟢 LOW RISK
<br><br>
Confidence : {probability:.2f}%
</div>
""",
                unsafe_allow_html=True
            )

        elif risk == "Medium":

            st.markdown(
                f"""
<div style="
background:#F59E0B;
padding:25px;
border-radius:18px;
text-align:center;
color:white;
font-size:28px;
font-weight:bold;">
🟠 MEDIUM RISK
<br><br>
Confidence : {probability:.2f}%
</div>
""",
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
<div style="
background:#DC2626;
padding:25px;
border-radius:18px;
text-align:center;
color:white;
font-size:28px;
font-weight:bold;">
🔴 HIGH RISK
<br><br>
Confidence : {probability:.2f}%
</div>
""",
                unsafe_allow_html=True
            )

        st.success("Prediction completed successfully.")

        st.dataframe(input_df, use_container_width=True)















       
elif selected == "Dataset Analytics":

    st.subheader("Dataset Analytics")

    risk = df["Infrastructure_Risk"].value_counts().reset_index()
    risk.columns = ["Risk", "Count"]

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            risk,
            names="Risk",
            values="Count",
            hole=0.55,
            color="Risk",
            color_discrete_map={
                "Low": "#22c55e",
                "Medium": "#f59e0b",
                "High": "#ef4444"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            title="Risk Distribution",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.bar(
            risk,
            x="Risk",
            y="Count",
            text="Count",
            color="Risk",
            color_discrete_map={
                "Low": "#22c55e",
                "Medium": "#f59e0b",
                "High": "#ef4444"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            title="Risk Count",
            showlegend=False,
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col3, col4 = st.columns(2)

    with col3:

        if "YEAR_BUILT_027" in df.columns:

            fig = px.histogram(
                df,
                x="YEAR_BUILT_027",
                nbins=35,
                title="Distribution of Year Built"
            )

            fig.update_layout(
                template="plotly_dark",
                height=420
            )

            st.plotly_chart(fig, use_container_width=True)

    with col4:

        if "ADT_029" in df.columns:

            fig = px.histogram(
                df,
                x="ADT_029",
                nbins=35,
                title="Average Daily Traffic"
            )

            fig.update_layout(
                template="plotly_dark",
                height=420
            )

            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(include=np.number)

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=False,
        aspect="auto",
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        template="plotly_dark",
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Numeric Feature Summary")

    st.dataframe(
        numeric_df.describe(),
        use_container_width=True
    )


elif selected == "Feature Importance":

    st.subheader("Top 15 Important Features")

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    ).head(15)

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        color="Importance",
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        template="plotly_dark",
        height=650,
        yaxis=dict(categoryorder="total ascending"),
        coloraxis_showscale=False,
        title="Most Important Features Used by XGBoost"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Feature Importance Table")

    importance["Importance"] = importance["Importance"].round(4)

    st.dataframe(
        importance,
        use_container_width=True
    )

    st.divider()

    st.info(
        "These features have the greatest influence on the XGBoost model while predicting infrastructure risk."
    )


elif selected == "About":

    st.subheader("About InfraRisk AI")

    st.markdown("""
### 🏗️ Project Description

InfraRisk AI is an AI-powered Infrastructure Risk Assessment System that predicts
bridge risk levels using Machine Learning.

The system analyzes bridge characteristics such as construction year, traffic,
dimensions, environmental factors and structural information to classify bridges
into *Low, **Medium, or **High* risk categories.

The project aims to assist infrastructure authorities in prioritizing bridge
maintenance and improving public safety through data-driven decision making.
""")

    st.divider()

    st.subheader("Technology Stack")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:
        st.success("🐍 Python")
        st.success("📊 Pandas")
        st.success("🔢 NumPy")

    with tech2:
        st.success("🤖 Scikit-learn")
        st.success("⚡ XGBoost")
        st.success("📈 Plotly")

    with tech3:
        st.success("🖥️ Streamlit")
        st.success("💾 Joblib")
        st.success("📋 Machine Learning")

    st.divider()

    st.subheader("Model Performance")

    performance = pd.DataFrame({
        "Model": [
            "Decision Tree",
            "Random Forest",
            "XGBoost"
        ],
        "Accuracy (%)": [
            69.4,
            73.4,
            74.7
        ]
    })

    st.dataframe(
        performance,
        use_container_width=True
    )

    st.divider()

    st.subheader("Developer")

    st.markdown("""
### 👨‍💻 Manish Parihar

B.Tech Computer Science Engineering

Machine Learning | Data Science | AI

This project demonstrates the application of Machine Learning for infrastructure risk assessment using the Florida Bridge Dataset and an XGBoost classification model.
""")

    st.divider()

st.markdown(
    """
---
<center>
<h4>🏗️ InfraRisk AI</h4>
<p>Made by <b>Manish Parihar</b></p>
</center>
""",
    unsafe_allow_html=True
)