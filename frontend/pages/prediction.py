import requests
import streamlit as st
import pandas as pd

api_url="http://127.0.0.1:8000/predict"
Demo_text="""
    California State Teachers Retirement System reduced its 
    Crocs stake by 2.9%, selling 1,722 shares and retaining 
    58,344 shares valued at approximately $7.0 million. 
    Institutional investors collectively own 93.44% of CROX.
""".strip()

def show_prediction():
    st.title("Live Financial Sentiment Prediction")
    st.write("""
    Enter a tweet, headline or text on stocks or economics.
""")
    st.divider()
    st.caption("Copy and Paste the Example and see what happens.")
    st.code(
        Demo_text.strip(),
        language=None,
        wrap_lines=True
    )

    if st.button("Put the Example into the Input Box"):
        st.session_state["financial_text"]=Demo_text

    text=st.text_area(
        "Financial comments",
        placeholder="Enter financial text here...",
        height=180,
        key="financial_text"
    )    

    if st.button("Predict Sentiment", type="primary"):
        if not text.strip():
            st.warning("Please enter a financial comment.")
            return 
        try:
            with st.spinner("Predicting"):
                response=requests.post(
                    api_url,
                    json={
                        "text": text
                    },
                    timeout=60,
                )

            if not response.ok:
                try:
                    error=response.json()
                    detail=error.get("detail", "Prediction failed.")
                except ValueError:
                    detail="Prediction failed."
                st.error(f"API error: {detail}")
                return

            result=response.json()

            result_df=pd.DataFrame([
                {
                    "Sentiment": result["sentiment"].capitalize(),
                    "Confidence": result["confidence"],
                    "Explanation": result["explanation"]
                }
            ])

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True,
                row_height=120,
                column_config={
                    "Sentiment": st.column_config.TextColumn(
            "Sentiment",
            width="small"
        ),
        "Confidence": st.column_config.TextColumn(
            "Confidence",
            width="small"
        ),
        "Explanation": st.column_config.TextColumn(
            "Explanation",
            width="large"
        ),
                }
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "Couldn't connect to FastAPI server."
            )

        except requests.exceptions.Timeout:
            st.error(
                "The prediction took too long. Try again with a shorter comment."
            )

        except requests.exceptions.RequestException as e:
            st.error(
                f"Request failed: {e}"
            )