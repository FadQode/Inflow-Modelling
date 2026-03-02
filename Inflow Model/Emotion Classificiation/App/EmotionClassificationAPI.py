from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load tokenizer & model dari local folder
MODEL_PATH = "./EkmanClassifier"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Emosi label Ekman
emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

# FastAPI init
app = FastAPI(title="Emotion Classification API")

# Request body model
class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_emotion(input: TextInput):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        top_class = torch.argmax(probs, dim=1).item()
        predicted_emotion = emotion_labels[top_class]
    
    return {
        "text": input.text,
        "predicted_emotion": predicted_emotion,
        "confidence_scores": {emotion_labels[i]: round(prob.item(), 4) for i, prob in enumerate(probs[0])}
    }
