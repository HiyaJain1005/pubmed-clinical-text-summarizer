import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from utils import (
    clean_text,
    preprocess_sentence,
    create_labels,
    get_columns
)

DATA_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

def build_data(df, article_col, summary_col, limit=500):
    X_text = []
    y = []

    df = df[[article_col, summary_col]].dropna().head(limit)

    for _, row in df.iterrows():
        article = clean_text(row[article_col])
        summary = clean_text(row[summary_col])

        sentences, labels = create_labels(article, summary)

        for s, l in zip(sentences, labels):
            ps = preprocess_sentence(s)
            if ps.strip():
                X_text.append(ps)
                y.append(l)

    return X_text, y

def main():
    print("Loading training data...")
    df = pd.read_csv(DATA_PATH)

    article_col, summary_col = get_columns(df)

    if not article_col or not summary_col:
        raise Exception("Column names not detected. Check CSV.")

    print("Using:", article_col, summary_col)

    X_text, y = build_data(df, article_col, summary_col, limit=500)

    print("Samples:", len(X_text))

    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_text, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    print("\nValidation Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("Model saved.")

    # ---------- TEST EVALUATION ----------
    print("\nRunning test evaluation...")

    test_df = pd.read_csv(TEST_PATH)
    test_df = test_df[[article_col, summary_col]].dropna().head(50)

    test_sentences = []
    test_labels = []

    for _, row in test_df.iterrows():
        sents, labels = create_labels(
            row[article_col],
            row[summary_col]
        )
        for s, l in zip(sents, labels):
            test_sentences.append(preprocess_sentence(s))
            test_labels.append(l)

    X_test = vectorizer.transform(test_sentences)
    y_test_pred = model.predict(X_test)

    print("Test Accuracy:", accuracy_score(test_labels, y_test_pred))

if __name__ == "__main__":
    main()