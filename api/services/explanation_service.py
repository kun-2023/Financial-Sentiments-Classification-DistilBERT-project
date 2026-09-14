from functools import lru_cache
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
from api.settings import llm_model_name



class ExplanationService:
    def __init__(self):
        self.tokenizer=self._load_tokenizer()
        self.model=self._load_model()

    def _load_tokenizer(self):
        return AutoTokenizer.from_pretrained(llm_model_name)

    def _load_model(self):
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32

        model=AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            dtype=dtype,
            device_map="auto"
        )

        model.eval()

        return model

    def generate_explanation(
            self,
            text:str,
            sentiment: str,
            confidence: float
    ) -> str:

        if not text or not text.strip():
            raise ValueError("Text can't be empty.")

        messages=[{
            "role": "system",
            "content": ("""
            You explain financial sentiment predictions clearly and concisely.
            Explain why the predicted sentiment is positive, negative, or neutral. 
            You will only do so strictly with the information provided by 
            texts and sentiment. Don't invent."""
            )
        },


        {
            "role": "user",
            "content": (
                f"Financial text: \n{text.strip()}\n\n"
                f"Predicted sentiment: {sentiment}\n"
                f"Confidence: {confidence:.2%}\n\n"
                "Explain the predicted sentiment in 2-3 sentences"
            )
        }
        ]

        inputs=self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )   

        inputs=inputs.to(self.model.device)    

        with torch.inference_mode():
            outputs=self.model.generate(
                inputs,
                max_new_tokens=30,
                do_sample=False,
            )

        generated_tokens=outputs[0][inputs.shape[-1]:]

        explanation=self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        if "." in explanation:
            return explanation.rsplit(".",1)[0]+"."
        return explanation.strip()

@lru_cache(maxsize=1)
def get_explanation_service() -> ExplanationService:
    return ExplanationService()

def generate_explanation(
        text: str,
        sentiment: str,
        confidence: float,
) -> str:
    
    service=get_explanation_service()

    return service.generate_explanation(
        text=text,
        sentiment=sentiment,
        confidence=confidence
    )
        
