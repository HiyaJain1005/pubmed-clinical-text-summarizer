import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import re
import nltk

from utils import preprocess_sentence

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="PubMed Summarizer", layout="wide")

MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

# ---------------------- NLTK SETUP (CACHED) ----------------------
@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

setup_nltk()

# ---------------------- SIMPLE SENTENCE SPLITTER ----------------------
def split_sentences(text):
    return re.split(r'(?<=[.!?]) +', text.strip())

# ---------------------- STYLING ----------------------
st.markdown("""
<style>
.stApp { background-color: #0b1220; color: #e5e7eb; }
.main-card { background: linear-gradient(135deg, #1e3a8a, #0f766e); padding: 28px; border-radius: 16px; margin-bottom: 24px; }
.main-title { font-size: 34px; font-weight: 600; color: #93c5fd; }
.main-sub { font-size: 14px; color: #cbd5f5; }
.block { background: #111827; padding: 18px; border-radius: 12px; border: 1px solid #1f2937; }
textarea { background-color: #0f172a !important; color: white !important; }
.stButton button { background: linear-gradient(90deg, #2563eb, #10b981); color: white; border-radius: 10px; }
.highlight { background-color: #22c55e; color: black; padding: 2px 6px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
st.markdown("""
<div class="main-card">
    <div class="main-title">PubMed Clinical Literature Summarizer</div>
    <div class="main-sub">
        ML-Based Extractive NLP using TF-IDF + Logistic Regression
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model files missing. Upload model.pkl and vectorizer.pkl in /models")
        st.stop()
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)

model, vectorizer = load_model()

# ---------------------- UTIL ----------------------
def get_scores(text):
    sents = split_sentences(text)
    proc = [preprocess_sentence(s) for s in sents]
    X = vectorizer.transform(proc)
    scores = model.predict_proba(X)[:, 1]
    return sents, scores

def summarize(text, ratio):
    sents, scores = get_scores(text)
    n = max(1, int(len(sents) * ratio))
    idx = sorted(np.argsort(scores)[-n:])
    summary = " ".join([sents[i] for i in idx])
    return summary, sents, scores, idx

def keywords(sent):
    words = re.findall(r'\b\w+\b', sent.lower())
    return list(dict.fromkeys([w for w in words if len(w) > 4]))[:3]

def confidence(scores):
    avg = np.mean(scores)
    return "High" if avg > 0.6 else "Moderate" if avg > 0.4 else "Low"

# ---------------------- TABS ----------------------
tab1, tab2, tab3, tab4 = st.tabs(["Home", "Summarizer", "Insights", "About"])

# ---------------------- HOME ----------------------
with tab1:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.subheader("Overview")
    st.write("ML-based extractive summarization for biomedical text using TF-IDF and Logistic Regression.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- SUMMARIZER ----------------------
with tab2:

    SAMPLE = """Biomedical research is growing rapidly. Manual analysis is time-consuming.
Machine learning enables automatic summarization. Extractive methods select key sentences.
TF-IDF with Logistic Regression is commonly used for sentence scoring."""

    if st.button("Load Sample Text"):
        st.session_state["input"] = SAMPLE

    col1, col2 = st.columns([2.5, 1])

    with col1:
        st.markdown("<div class='block'>", unsafe_allow_html=True)
        text = st.text_area("Input Text", height=280, key="input")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='block'>", unsafe_allow_html=True)
        mode = st.selectbox("Summary Mode", ["Concise", "Balanced", "Detailed"])
        ratio = {"Concise": 0.2, "Balanced": 0.3, "Detailed": 0.5}[mode]
        file = st.file_uploader("Upload", type=["txt", "pdf"])
        st.markdown("</div>", unsafe_allow_html=True)

    if file:
        if file.type == "application/pdf":
            import PyPDF2
            reader = PyPDF2.PdfReader(file)
            text = " ".join([p.extract_text() or "" for p in reader.pages])
        else:
            text = file.read().decode("utf-8")
        st.session_state["input"] = text

    if st.button("Generate Summary"):
        text = st.session_state.get("input", "")

        if not text.strip():
            st.warning("Enter text first")
            st.stop()

        with st.spinner("Generating summary..."):
            summary, sents, scores, idx = summarize(text, ratio)

        st.session_state["analysis"] = (sents, scores)

        st.markdown("<div class='block'>", unsafe_allow_html=True)
        st.subheader("Summary")
        st.write(summary)

        c1, c2, c3 = st.columns(3)
        c1.metric("Original Words", len(text.split()))
        c2.metric("Summary Words", len(summary.split()))
        c3.metric("Confidence", confidence(scores))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='block'>", unsafe_allow_html=True)
        st.subheader("Highlighted Sentences")

        out = ""
        for i, s in enumerate(sents):
            out += f"<span class='highlight'>{s}</span> " if i in idx else s + " "

        st.markdown(out, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- INSIGHTS ----------------------
with tab3:
    st.markdown("<div class='block'>", unsafe_allow_html=True)

    if "analysis" not in st.session_state:
        st.info("Run summarization first")
        st.stop()

    sents, scores = st.session_state["analysis"]

    df = pd.DataFrame({
        "Sentence": sents,
        "Score": scores
    }).sort_values("Score", ascending=False)

    st.dataframe(df)
    st.bar_chart(df.set_index("Sentence").head(10))

    sel = st.selectbox("Inspect Sentence", sents)
    i = sents.index(sel)

    st.write("Score:", round(float(scores[i]), 3))
    st.write("Keywords:", keywords(sel))

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- ABOUT ----------------------
with tab4:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.write("""
TF-IDF + Logistic Regression is used to score sentences.
Top-ranked sentences are selected to generate extractive summaries for biomedical text.
""")
    st.markdown("</div>", unsafe_allow_html=True)
