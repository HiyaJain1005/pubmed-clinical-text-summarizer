import joblib
import numpy as np
import os

from utils import split_sentences, preprocess_sentence

MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise Exception("Train model first")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer

def summarize(text, num_sentences=3):
    sentences = split_sentences(text)

    if len(sentences) <= num_sentences:
        return text

    model, vectorizer = load_model()

    processed = [preprocess_sentence(s) for s in sentences]
    X = vectorizer.transform(processed)

    scores = model.predict_proba(X)[:, 1]

    top_idx = np.argsort(scores)[-num_sentences:]
    top_idx = sorted(top_idx)

    summary = " ".join([sentences[i] for i in top_idx])
    return summary