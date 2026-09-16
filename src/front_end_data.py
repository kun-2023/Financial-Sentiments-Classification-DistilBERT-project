import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from src.config import config
from src.inference import load_model, predict_sentiment
from src.llm_explainer import load_llm, explain_sentiment
from pathlib import Path

demo_sentences=[
    "Adobe said artificial intelligence remains a key growth area, with AI-first ending annual recurring revenue exceeding $650 million and increasing more than 150% from a year earlier. The company also agreed to acquire Topaz Labs, subject to regulatory clearance.",
    "Lululemon’s newest bear sees more pain ahead after 81% rout.The entire athletic complex is in free fall, but the negativity right now is probably when there’s the most opportunity",
    "Jim Cramer declared LULU a 'thoroughly broken stock,' down 52% year to date and 77% over five years, with no compelling reason to buy.",
    "PayPal stock gains as turnaround strategy takes shape after failed buyout",
    "Palantir Technologies (NASDAQ: PLTR) received a higher Wall Street price target after DA Davidson reaffirmed its bullish stance on the software and artificial intelligence company.",
    "Microsoft Stock: Is Now the Right Time to Buy or Should You Hold Off?",
    "Wall Street has a Strong Buy consensus rating on AMD stock, with the average price target indicating 25.1% upside potential.",
    "Micron Stock Can Absorb a 50% Earnings Reset—Not a 75% One",
]

# Load saved test metrics
def load_test_metrics():
    """
    Load test metrics generated during model training;
    """
    with open(config["path"]["test_metrics"], "r") as f:
        metrics=json.load(f)
        return metrics


def confusion_matrix_normalized(model):
    """
    Evaluate the deployed model on the held-out test set and return a row-normalized confusion matrix;
    """
    test_df=pd.read_csv(
        config["path"]["cleaned_split_test"]
    )

    label_encoder=joblib.load(config["path"]["label_encoder"])

    texts=test_df["cleaned_text"].to_list()

    # Convert numeric labels back to sentiment labels
    y_true=label_encoder.inverse_transform(test_df["senti_label"].to_numpy())

    # Predict entire test set
    results=model(texts, 
                  batch_size=config["distilbert"]["eval_batch_size"],
                  truncation=True)

    y_pred=[result["label"] for result in results]
    labels=label_encoder.classes_

    cm=confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    cm_df=pd.DataFrame(cm, index=labels, columns=labels)
    cm_df_output_dir=Path(config["path"]["cm"])
    cm_df.to_csv(
            cm_df_output_dir, index=True
        )
    return cm_df

# Generate demo predictions
def build_demo_dataframe(sentiment_model, llm_model, llm_tokenizer):
    """
    run sentiment classification and LLM explanation for the demo
    """
    rows=[]
    for text in demo_sentences:
        if not text.strip():
            continue
        prediction=predict_sentiment(text=text, model=sentiment_model)
        explanation=explain_sentiment(
            text=text,
            sentiment=prediction["sentiment"],
            confidence=prediction["confidence"],
            model=llm_model,
            tokenizer=llm_tokenizer
        )

        rows.append({
            "Text": text,
            "Sentiment": prediction["sentiment"],
            "Probability": str(np.round(prediction["confidence"]*100,2))+"%",
            "Explanation": explanation
        })

    demo_df=pd.DataFrame(rows)
    return demo_df

# sum up
def prepare_frontend_data():
    # load champion model
    sentiment_model=load_model(alias="champion")

    llm_model, llm_tokenizer=load_llm()

    metrics=load_test_metrics()

    confusion_matrix_df=confusion_matrix_normalized(sentiment_model)

    demo_df=build_demo_dataframe(
        sentiment_model=sentiment_model,
        llm_model=llm_model,
        llm_tokenizer=llm_tokenizer
    )
    # save demo_df to nlp_classification_model folder
    demo_df_output_dir=Path(config["path"]["demo_data"])

    demo_df.to_csv(
        demo_df_output_dir, index=False
    )

    return {
        "metrics": metrics,
        "confusion_matrix": confusion_matrix_df,
        "demo": demo_df
    }

if __name__=="__main__":
    data=prepare_frontend_data()
    print("\nMetrics:")
    print(data["metrics"])

    print("\nNormalized confusion matrix:")
    print(data["confusion_matrix"])

    print("\nDemo predictions:")
    print(data["demo"])