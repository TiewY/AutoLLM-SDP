import streamlit as st
import json
from utils import extract_features, classify_project

st.set_page_config(page_title="AutoLLM Demo", layout="wide")

st.title("🚀 AutoLLM: Model Selection Framework for Software Defect Prediction")

st.info("""
🧠 AutoLLM is NOT a defect prediction model.

It is a meta-learning framework that automatically selects 
the best Software Defect Prediction (SDP) model 
based on learned experimental knowledge.
""")

# ============================
# Upload Section
# ============================
st.header("📂 Upload Your Project Code")

uploaded_file = st.file_uploader("Upload a code file", type=["py", "java", "txt"])

if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")

    # ============================
    # Feature Extraction
    # ============================
    st.header("🔍 Feature Extraction")

    features = extract_features(content)

    st.write(features)

    # ============================
    # AutoLLM Decision Engine
    # ============================
    st.header("🧠 AutoLLM Decision Engine")

    project_type = classify_project(features)

    st.write(f"📌 Project Type: **{project_type.upper()}**")

    # Load knowledge base
    with open("model_knowledge.json") as f:
        knowledge = json.load(f)

    recommendation = None
    for rule in knowledge["rules"]:
        if rule["project_type"] == project_type:
            recommendation = rule["recommendation"]
            break

    # ============================
    # Recommendation Output
    # ============================
    st.header("🎯 Recommended SDP Pipeline")

    st.success(f"Model: {recommendation['model']}")
    st.write("Embedding:", recommendation["embedding"])
    st.write("Optimizer:", recommendation["optimizer"])

    st.subheader("📈 Expected Performance")
    st.write("F1 Score:", recommendation["expected_f1"])
    st.write("MCC:", recommendation["expected_mcc"])

    st.subheader("🧠 Explanation")
    st.write(recommendation["reason"])

    st.progress(100)
    st.success("✅ AutoLLM Analysis Complete!")