# 🧠 PubMed Clinical Literature Summarizer  
### An Explainable ML-Based Extractive NLP System for Biomedical Text

---

## 📌 Overview

Biomedical research is growing at an exponential rate, making it difficult for clinicians and researchers to quickly extract meaningful insights from large volumes of text.

This project presents an **extractive text summarization system** that uses **Machine Learning (TF-IDF + Logistic Regression)** to identify and rank important sentences from biomedical literature.

The system generates concise summaries while maintaining interpretability through sentence-level scoring and highlighting.

---

## 🚀 Features

- Extractive text summarization  
- Sentence importance scoring (Explainable AI)  
- Highlighted key sentences  
- Interactive web interface using Streamlit  
- Supports text input and PDF uploads  
- Adjustable summary length (Concise / Balanced / Detailed)  

---

## 🧠 Methodology


The system follows a structured NLP pipeline:
Input Text
↓
Sentence Tokenization
↓
Text Preprocessing (Stopword Removal, Lemmatization)
↓
TF-IDF Vectorization
↓
Logistic Regression Classification
↓
Sentence Scoring
↓
Top Sentence Selection
↓
Final Summary Output


---

## 🤖 Model Details

### TF-IDF Vectorization
- Converts text into numerical feature vectors  
- Captures importance of words in context  

### Logistic Regression
- Classifies sentences based on importance  
- Outputs probability scores for ranking  

---

---

## 🤖 Model Details

### TF-IDF Vectorization
- Converts text into numerical feature vectors  
- Captures importance of words in context  

### Logistic Regression
- Classifies sentences based on importance  
- Outputs probability scores for ranking  

---

## 📊 Dataset

- **PubMed Dataset**
- Contains biomedical research articles paired with abstracts  
- Abstracts are used as reference summaries for training and evaluation  

---

## 🖥️ Application (Streamlit UI)

The project includes an interactive web application where users can:

- Input biomedical text  
- Upload PDF or text files  
- Generate summaries instantly  
- View highlighted important sentences  
- Analyze sentence scores and keywords  

---

## 📂 Project Structure
pubmed_summarizer/
│
├── app.py # Streamlit application
├── train_model.py # Model training script
├── utils.py # Text preprocessing & helpers
├── models/
│ ├── model.pkl
│ └── vectorizer.pkl
├── data/ # Dataset (train/test/val)
└── README.md


---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/pubmed-summarizer.git
cd pubmed-summarizer

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt

streamlit run app.py
---
📈 Results
Generates concise summaries from long biomedical text
Highlights key sentences for interpretability
Maintains essential context and meaning
⚠️ Limitations
Extractive only (does not generate new sentences)
Limited contextual understanding
Summary coherence depends on sentence order
Performance depends on dataset quality
🔮 Future Scope
Abstractive summarization using transformers (BERT, T5)
Multilingual biomedical summarization
Integration with clinical decision-support systems
Domain-specific fine-tuning
🧩 Tech Stack
Python
NLTK
Scikit-learn
Streamlit
Pandas, NumPy

Author
Hiya Jain
