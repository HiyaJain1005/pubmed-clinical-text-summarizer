import re

# ---------------------- SENTENCE SPLIT ----------------------
def split_sentences(text):
    return re.split(r'(?<=[.!?]) +', text.strip())

# ---------------------- PREPROCESS ----------------------
def preprocess_sentence(sentence):
    sentence = sentence.lower()

    # Remove special characters
    sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)

    # Simple word split (no NLTK)
    words = sentence.split()

    # Basic stopwords (minimal set)
    stopwords = {
        "the", "is", "in", "and", "to", "of", "a", "for",
        "on", "with", "as", "by", "an", "be", "are", "at"
    }

    words = [w for w in words if w not in stopwords]

    return " ".join(words)
