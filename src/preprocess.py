import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import config

label_map={
    0: "negative",
    1: "positive",
    2: "neutral",
}

# Load raw data

def load_raw_data():
    train=pd.read_csv(config["path"]["raw_train"])
    valid=pd.read_csv(config["path"]["raw_valid"])
    return train, valid

# clean text

def clean_text(text):
    # remove URLs
    text=re.sub(r"https?://\S+", "", text)

    # Remove stock tocker symbols
    text=re.sub(
        r"\$[A-Za-z]+\s*[-,:|]?\s*",
        "", text
    )

    # remove <br> tags
    text=re.sub(r"<br\s*/?>", "", text)

    # remove extra spaces
    text=re.sub(r"\s+", " ",text).strip()

    return text


# clean dataframe

def preprocess_dataframe(df):
    df=df.copy()
    df["cleaned_text"]=df.text.apply(clean_text)
    df["sentiments"]=df.label.map(label_map)
    return df

# Split valid dataset into valid and test
def create_splits(train, valid):
    train=train[["cleaned_text", "sentiments"]].copy()
    valid, test=train_test_split(valid[["cleaned_text", "sentiments"]], 
                                 test_size=0.5,
                                 random_state=config["random_state"]["random_state"],
                                stratify=valid["sentiments"]
                                 )
    return train, valid, test

# Encode sentiment labels
def encode_labels(train, valid, test):
    label_encoder=LabelEncoder()
    train["senti_label"]=label_encoder.fit_transform(train["sentiments"])
    valid["senti_label"]=label_encoder.transform(valid["sentiments"])
    test["senti_label"]=label_encoder.transform(test["sentiments"])
    return train, valid, test, label_encoder

    
def save_data(cleaned_train, 
              cleaned_valid, 
              split_train,
              split_valid,
              split_test,
              label_encoder):

    # create directories if they don't exist
    config["path"]["cleaned_data_train"].parent.mkdir(parents=True, exist_ok=True)
    config["path"]["cleaned_split_train"].parent.mkdir(parents=True, exist_ok=True)

    # Save cleaned data
    cleaned_train[["cleaned_text", "sentiments"]].to_csv(config["path"]["cleaned_data_train"],index=False)
    cleaned_valid[["cleaned_text", "sentiments"]].to_csv(config["path"]["cleaned_data_valid"],index=False)

    # Save cleaned_split_data
    split_train[["cleaned_text", "senti_label"]].to_csv(config["path"]["cleaned_split_train"], index=False)
    split_valid[["cleaned_text", "senti_label"]].to_csv(config["path"]["cleaned_split_valid"], index=False)
    split_test[["cleaned_text", "senti_label"]].to_csv(config["path"]["cleaned_split_test"], index=False)

    # Save Label ENcoder
    encoder_path=config["path"]["label_encoder"]
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(label_encoder, encoder_path)

# Run preproessing pipeline

def run_preprocessing():
    # 1_load original raw datasets
    raw_train, raw_valid=load_raw_data()
    # 2_remove missing texts/labels
    raw_train=raw_train.dropna(subset=["text", "label"])
    raw_valid=raw_valid.dropna(subset=["text", "label"])
    #3_clean original dataset
    cleaned_train=preprocess_dataframe(raw_train)    
    cleaned_valid=preprocess_dataframe(raw_valid)
    cleaned_train.dropna(inplace=True)
    cleaned_valid.dropna(inplace=True)
    #4_remove rows whose text become empty after cleaning
    cleaned_train=cleaned_train[cleaned_train["cleaned_text"].str.strip().ne("")].copy()
    cleaned_valid=cleaned_valid[cleaned_valid["cleaned_text"].str.strip().ne("")].copy()
    #5_create final train/test/valid datasets
    split_train, split_valid, split_test=create_splits(cleaned_train, cleaned_valid)
    #6_Encode labels
    (split_train, split_valid, split_test, label_encoder)=encode_labels(
        split_train, split_valid, split_test)
    #7_save_both_stages
    save_data(
        cleaned_train=cleaned_train,
        cleaned_valid=cleaned_valid,
        split_train=split_train,
        split_valid=split_valid,
        split_test=split_test,
        label_encoder=label_encoder
    )

    return split_train, split_valid, split_test

if __name__=="__main__":
    run_preprocessing()






