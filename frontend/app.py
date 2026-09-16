import streamlit as st

from pages.overview import show_overview
from pages.demo import show_demo
from pages.prediction import show_prediction
from pages.metrics import show_metrics


st.set_page_config(
    page_title="Financial Sentiment Analysis",
    page_icon="",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {
        width: 80%;
        max-width: 80%;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True
)
overview_page=st.Page( show_overview, title="Overview", default=True)
metrics_page=st.Page(show_metrics, title="Metrics")
demo_page=st.Page(show_demo, title="Demo")
prediction_page=st.Page(show_prediction,  title="Prediction")

pg=st.navigation([overview_page, metrics_page, demo_page, prediction_page])

pg.run()

