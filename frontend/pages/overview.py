from pathlib import Path
import json
import streamlit as st
import pandas as pd

project_root=Path(__file__).resolve().parents[2]
test_metrics_path=(project_root/"outputs"/"nlp_classification_model"/"test_metrics.json")
def show_overview():
    st.title("Financial Sentiment Analysis")
    st.subheader("Project Overview")
    st.write(
        """
    This project is an end-to-end machine learning and 
    large language model powered AI application.
    
    **Model Input:** Financial-related text such as tweets or headlines.

    **Model Outputs:**
    - Predicted Sentiment;
    - Predicted Probability;
    - Expalantion on why it's the predicted sentiment.
    """)

    st.subheader("Models")
    model_table = pd.DataFrame({
    "Model Name": [
        "DistilBERT",
        "SmolLM2-360M-Instruct"
    ],
    "Model Type": [
        "Encoder-only Transformer",
        "Small generative language model"
    ],
    "Usage": [
        "Financial sentiment classification",
        "Generate human-readable explanations"
    ]
})
    st.table(model_table)

    st.subheader("Tech Stacks")
    st.write("""
        Python, pandas, NumPy, scikit-learn, 
    PyTorch, HuggingFace Transformers(DistilBert), SmoILM2-360M-Instruct,
    MLflow, DVC, Git/GitHub, FastAPI, Streamlit, pytest, Docker, AWS, ECR,
    ECS, logging, GitHub Actions and more.
"""
    )

    