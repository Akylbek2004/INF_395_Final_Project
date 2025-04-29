import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stopword = stopwords.words('english')
punc = string.punctuation

def clean_text(text):
    text = re.sub('<.*?>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.translate(str.maketrans('', '', punc))
    text = " ".join([word for word in text.split() if word.lower() not in stopword])
    return text.lower()

model = joblib.load("rf_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Sentiment Classifier", page_icon="🎬")
st.markdown("""
    <style>
    body {
        background-color: #0f1117;
        color: white;
    }
    .main {
        background-color: #0f1117;
    }
    .box {
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        font-weight: bold;
        font-size: 18px;
        text-align: center;
    }
    .positive {
        background-color: #14532d;
        color: white;
    }
    .negative {
        background-color: #7f1d1d;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 style="text-align: center;">🎬 Sentiment Classifier</h1>', unsafe_allow_html=True)
st.write("Enter the text of the review, and the model will tell you whether it is positive or negative.")

user_input = st.text_area("Enter a review:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter the text.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        sentiment = label_encoder.inverse_transform([prediction])[0]

        css_class = "positive" if sentiment.lower() == "positive" else "negative"
        emoji = "✅" if sentiment.lower() == "positive" else "❌"
        st.markdown(f'<div class="box {css_class}">{emoji} This is a <strong>{sentiment}</strong> review!</div>', unsafe_allow_html=True)
