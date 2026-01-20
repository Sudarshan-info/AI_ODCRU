from fastapi import FastAPI
from pydantic import BaseModel
import os
import joblib
import numpy as np
import tensorflow as tf

from src.preprocessing import clean_text


app = FastAPI(title="Sentiment Analysis API")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


vectorizer = joblib.load("models_saved/tfidf_vectorizer.pkl")
nb_model = joblib.load("models_saved/naive_bayes.pkl")
dl_model = tf.keras.models.load_model("models_saved/deep_learning.h5")

label_map = {0:"negative",1:"neutral",2:"positive"}

class TextInput(BaseModel):
    text: str
    model: str = "naive_bayes"  

@app.get("/")
def health_check():
    return {"status": "API running"}

@app.post("/predict")
def predict_sentiment(data: TextInput):
    cleaned = clean_text(data.text)
    vec = vectorizer.transform([cleaned])

    if data.model == "deep_learning":
        pred = dl_model.predict(vec.toarray())
        sentiment = label_map[int(np.argmax(pred))]
    else:
        pred = nb_model.predict(vec)
        sentiment = label_map[int(pred[0])]

    return {
        "input_text": data.text,
        "model_used": data.model,
        "predicted_sentiment": sentiment
    }
