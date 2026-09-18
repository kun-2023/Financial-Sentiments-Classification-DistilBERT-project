from src.preprocess import clean_text, preprocess_dataframe
import pandas as pd

def test_remove_url():
    text="Adobe shares rose https://example.com"
    result=clean_text(text)
    assert result=="Adobe shares rose"

def test_remove_ticker():
    text="$ADBE Adobe shares rose today"    
    result=clean_text(text)
    assert result=="Adobe shares rose today"

def test_remove_multiple_tickers():
    text="$AAPL $MSFT: Apple and Microsoft shares rose"

    result=clean_text(text)
    assert "$AAPL" not in result
    assert "$MSFT" not in result
    assert result=="Apple and Microsoft shares rose"

def test_remove_br_tag():
    text="$ADBE Adobe shares rose <br> after earnings"
    result=clean_text(text)

    assert "<br>" not in result
    assert result=="Adobe shares rose after earnings"

def test_remove_extra_spaces():
    text="Adobe   shares   rose"
    result=clean_text(text)
    assert result=="Adobe shares rose"

def test_preprocess_dataframe():
    df=pd.DataFrame({
        "text": [
            "$ADBE Adobe shares rose https://example.com",
            "Market remains weak"
        ],
        "label": [
            1,0
        ]
    })

    result=preprocess_dataframe(df)

    assert "cleaned_text" in result.columns
    assert "sentiments" in result.columns

    assert result.loc[0, "cleaned_text"]=="Adobe shares rose"
    assert result.loc[0, "sentiments"]=="positive"

    assert result.loc[1, "cleaned_text"]=="Market remains weak"
    assert result.loc[1, "sentiments"]=="negative"
    