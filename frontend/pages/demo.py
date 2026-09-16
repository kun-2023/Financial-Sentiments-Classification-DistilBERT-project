from pathlib import Path
import pandas as pd
import streamlit as st

project_root=Path(__file__).resolve().parents[2]
demo_data_path=(
    project_root/"outputs/nlp_classification_model/demo_df.csv"
)

def show_demo():
    st.title("Demo Examples")
    st.write("""The following shows how the app 
    returns outputs based on the texts provided.""")

    demo_df=pd.read_csv(demo_data_path)
    #table_height=40+len(demo_df)*100
    st.dataframe(
        demo_df,
        width="stretch",
        row_height=100,
        height=540,
        hide_index=True,
        column_config={
            "Text":st.column_config.TextColumn(
                "Financial Text",
                width="large"
            ),
            "Explanation":st.column_config.TextColumn(
                "Sentiment Explanation",
                width="large"
            ),
        }
    )