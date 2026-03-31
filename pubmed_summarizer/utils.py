import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def split_sentences(text):
    text = clean_text(text)
    return sent_tokenize(text)

def preprocess_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-zA-Z0-9\s]", "", sentence)
    words = word_tokenize(sentence)
    words = [w for w in words if w not in STOP_WORDS]
    return " ".join(words)

def get_columns(df):
    # Modify if needed after checking CSV
    article_col = None
    summary_col = None

    for col in df.columns:
        if col.lower() in ["article", "body_text", "sections", "text"]:
            article_col = col
        if col.lower() in ["abstract", "summary"]:
            summary_col = col

    return article_col, summary_col

def overlap_score(sentence, summary):
    s1 = set(preprocess_sentence(sentence).split())
    s2 = set(preprocess_sentence(summary).split())

    if not s1 or not s2:
        return 0

    return len(s1 & s2) / len(s1)

def create_labels(article, summary, threshold=0.2):
    sentences = split_sentences(article)
    labels = []

    for s in sentences:
        score = overlap_score(s, summary)
        labels.append(1 if score >= threshold else 0)

    return sentences, labels