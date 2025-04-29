import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import zipfile

# Download NLTK stopwords
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
punc = string.punctuation

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import re

# Load IMDB dataset
df = pd.read_csv("IMDB Dataset.csv")

# --- Text Preprocessing Functions ---

# Remove HTML tags
def remove_html_tags(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'', text)
df['review'] = df['review'].apply(remove_html_tags)

# Remove URLs
def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)
df['review'] = df['review'].apply(remove_url)

# Remove punctuation
def remove_punc(text):
    return text.translate(str.maketrans('', '', punc))
df['review'] = df['review'].apply(remove_punc)

# Dictionary for converting chat abbreviations to full form
chat_words = {
    # [abbreviation]: "full form"
    "LOL": "Laughing Out Loud",
    "BRB": "Be Right Back",
    "IMO": "In My Opinion",
    "WTF": "What The F...",
    "TTYL": "Talk To You Later",
    "ILY": "I Love You",
    "BFF": "Best Friends Forever",
    "JK": "Just Kidding",
    "IDC": "I Don't Care",
    # ... add more if needed
}

# Replace chat abbreviations with full words
def chat_conversion(text):
    new_text = []
    for i in text.split():
        if i.upper() in chat_words:
            new_text.append(chat_words[i.upper()])
        else:
            new_text.append(i)
    return " ".join(new_text)
df['review'] = df['review'].apply(chat_conversion)

# Remove stopwords
stopword = stopwords.words('english')
def remove_stopwords(text):
    new_text = [word for word in text.split() if word not in stopword]
    return " ".join(new_text)
df['review'] = df['review'].apply(remove_stopwords)

# Encode sentiment labels: 'positive' → 1, 'negative' → 0
label_encoder = LabelEncoder()
df['sentiment'] = label_encoder.fit_transform(df['sentiment'])

# --- Train/Test Split ---

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

X_train_text = train_data['review']
X_test_text = test_data['review']
y_train = train_data['sentiment']
y_test = test_data['sentiment']

# --- Vectorization & Model Training ---

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)

# Evaluate model
y_pred = rf_model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Save model and vectorizer for reuse (e.g., in Streamlit) ---
import joblib
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
