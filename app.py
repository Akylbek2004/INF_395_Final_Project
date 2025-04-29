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

# Download stopwords from NLTK
nltk.download('stopwords')

# Load English stopwords and punctuation
stopword = stopwords.words('english')
punc = string.punctuation

# Function to clean input text
def clean_text(text):
    text = re.sub('<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = text.translate(str.maketrans('', '', punc))  # Remove punctuation
    text = " ".join([word for word in text.split() if word.lower() not in stopword])  # Remove stopwords
    return text.lower()

# Load the trained model, vectorizer, and label encoder
model = joblib.load("rf_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Configure Streamlit page
st.set_page_config(page_title="Sentiment Classifier", page_icon="🎬")

# Apply custom CSS styling for dark theme and sentiment boxes
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

# Header and description
st.markdown('<h1 style="text-align: center;">🎬 Sentiment Classifier</h1>', unsafe_allow_html=True)
st.write("Enter the text of the review, and the model will tell you whether it is positive or negative.")

# User input text box
user_input = st.text_area("Enter a review:")

# Handle prediction logic when button is clicked
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter the text.")
    else:
        # Clean, transform and predict
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        sentiment = label_encoder.inverse_transform([prediction])[0]

        # Apply different style for positive or negative prediction
        css_class = "positive" if sentiment.lower() == "positive" else "negative"
        emoji = "✅" if sentiment.lower() == "positive" else "❌"
        st.markdown(f'<div class="box {css_class}">{emoji} This is a <strong>{sentiment}</strong> review!</div>', unsafe_allow_html=True)
