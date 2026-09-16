import torch
import re
from transformers import (AutoTokenizer, AutoModelForCausalLM)
from src.config import config

def load_llm():
    """
    Load the language model used to explain sentiment predictions
    """
    model_id=config["llm"]["model_id"]
    device=torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dtype=(
        torch.float16 if torch.cuda.is_available() else torch.float32
    )

    tokenizer=AutoTokenizer.from_pretrained(model_id)
    model=AutoModelForCausalLM.from_pretrained(model_id,  dtype=dtype)

    model.to(device)
    model.eval()

    print(f"Loaded explanation model: {model_id}")
    print(f"Device: {device}")
    return model, tokenizer

def explain_sentiment(
        text,
        sentiment,
        confidence,
        model,
        tokenizer
):
    """
    Generate a short explanation for a sentiment prediction. The LLM explains the prediction using only information contained in the supplied financial text.
    """
    confidence_pct=confidence*100
    messages=[
        {
            "role": "system",
            "content": (
                "You explain financial sentiment classifications. "
                "Explain which information in the financial text supports the predicted sentiment. "
                "Use only information contained in the text. Don't invent facts and keep explanation concise. "
            )
        },
        {
            "role": "user",
            "content": (
                f"Financial text:\n{text}\n\n"
                f"predicted sentiment: {sentiment}\n"
                f"Confidence: {confidence_pct:.1f}%\n\n"
                "Explain the sentiment in one complete sentence of no more than 40 words and end with a period."
            )
        }
    ]

    inputs=tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )    

    inputs=inputs.to(model.device)

    with torch.no_grad():
        output_ids=model.generate(
            **inputs,
            max_new_tokens=config["llm"]["max_new_tokens"],
            do_sample=config["llm"]["do_sample"],
            pad_token_id=tokenizer.eos_token_id
        )
    # keep only newly generated tokens, not the original prompt.
    generated_ids=output_ids[0, inputs["input_ids"].shape[-1]:]
    explanation=tokenizer.decode(
        generated_ids, 
        skip_special_tokens=True).strip()

    match=re.search(
        r"\.(?=\s+[A-Z]|$)", explanation
    )

    if match:
        return explanation[:match.end()].strip()

    return (
        f"The predicted sentiment is {sentiment} based on the "
        "information in the provided financial text."
    )