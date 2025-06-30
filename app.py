from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from fastapi.responses import JSONResponse
# Initialize FastAPI
app = FastAPI()

cache = "/app/hf_cache"
os.makedirs(cache, exist_ok=True)
os.environ["HF_HOME"] = cache
os.environ["TRANSFORMERS_CACHE"] = cache
os.environ["XDG_CACHE_HOME"] = cache

from transformers import AutoTokenizer 
# Load GRU model and tokenizer
gru_model = tf.keras.models.load_model('hs_gru.h5')
with open('tokenizerpkl_gru.pkl', 'rb') as f:
    gru_tokenizer = pickle.load(f)
gru_maxlen = 100

# Load RoBERTa model
# Load RoBERTa model
roberta_model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
if roberta_tokenizer.pad_token is None:
    roberta_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_model_name)
roberta_model.resize_token_embeddings(len(roberta_tokenizer))

#load toxigen-hatebert model
toxigen_model_name = "tomh/toxigen_roberta"
toxigen_tokenizer = AutoTokenizer.from_pretrained(toxigen_model_name)
if toxigen_tokenizer.pad_token is None:
    toxigen_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
toxigen_model = AutoModelForSequenceClassification.from_pretrained(toxigen_model_name)
toxigen_model.resize_token_embeddings(len(toxigen_tokenizer))

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic input model
class TextInput(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r") as f:
        return f.read()

@app.get("/health")
def health_check():
    return {"message": "Hate Speech Detection API is running!"}

@app.post("/predict")
def predict_ensemble(input: TextInput):
    try:
        text = input.text
        # print(f"Received input: {input.text}")

        # ----- GRU Prediction -----
        seq = gru_tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=gru_maxlen, padding='post')
        gru_prob = float(gru_model.predict(padded)[0][0])

        # ----- RoBERTa Prediction -----
        inputs_roberta = roberta_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits_roberta = roberta_model(**inputs_roberta).logits
            probs_roberta = torch.nn.functional.softmax(logits_roberta, dim=1)
            roberta_prob = float(probs_roberta[0][1].item())

        # -----toxigen -hatebert Prediction -----
        inputs_toxigen = toxigen_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits_toxigen = toxigen_model(**inputs_toxigen).logits
            probs_toxigen = torch.nn.functional.softmax(logits_toxigen, dim=1)
            toxigen_prob = float(probs_toxigen[0][1].item())

        # ----- Weighted Ensemble -----
        final_score = (0.3 * gru_prob) + (0.4 * roberta_prob) + (0.3 * toxigen_prob)
        label = "Hate Speech" if final_score > 0.5 else "Not Hate Speech"

        return {
            # "text": text,
            "gru_prob": round(gru_prob, 4),
            "roberta_prob": round(roberta_prob, 4),
            "toxigen_prob": round(toxigen_prob, 4),
            "final_score": round(final_score, 4),
            "prediction": label
        }

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": str(e)})
