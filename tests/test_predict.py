import requests
api_url="http://localhost:8000/predict"

def test_predict():
    response=requests.post(
        api_url, 
        json={
            "text": "Adobe shares rose after reporting stronger revenue."
        },
        timeout=120
    )

    assert response.status_code==200

    result=response.json()

    assert "sentiment" in result
    assert "confidence" in result
    assert "explanation" in result

    assert result["sentiment"] in [
        "negative", "neutral", "positive"
    ]

    assert isinstance(result["confidence"], str)
    assert result["confidence"].endswith("%")
    assert isinstance(result["explanation"], str)
    assert result["explanation"].strip() !=""