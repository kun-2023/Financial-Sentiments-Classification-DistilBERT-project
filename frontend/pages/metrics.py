import streamlit as st
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


project_root=Path(__file__).resolve().parents[2]
test_metrics_path=(project_root/"outputs"/"nlp_classification_model"/"test_metrics.json")
cm_path=(project_root/"outputs"/"nlp_classification_model"/"cm.csv")


def show_metrics():
    st.title("Model DistilBERT Test Metrics")
    with open(test_metrics_path, "r") as file:
        metrics=json.load(file)
    col1, col2, col3, col4=st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("Precision", f"{metrics['precision']:.2%}")
    col3.metric("Recall", f"{metrics['recall']:.2%}")
    col4.metric("F1", f"{metrics['f1']:.2%}")

    st.divider()

    st.subheader("Model DistilBERT Normalized Confusion Matrix")
    st.write("""
    Normalized Confusion Matrix shows Percentage of observed 
    values(in rows) to be predicted in terms of each 
    category(in columns). In each category, the model had correctly
    predicted sentiments in high percentage of 82% (negative), 
    90% (neutral) and 77% (positive). 
    """)
    cm_df=pd.read_csv(cm_path, index_col=0)
    fig,ax=plt.subplots(figsize=(10,10))
    display=ConfusionMatrixDisplay(
        confusion_matrix=cm_df.values,
        display_labels=cm_df.columns
    )
    display.plot(ax=ax, values_format=".2f", text_kw={"fontsize": 30})

    ax.set_title("DistilBERT Confusion Matrix")

    left, center, right=st.columns([1,4,1])
    with center:
        st.pyplot(fig,
                  use_container_width=True)
    plt.close(fig)        