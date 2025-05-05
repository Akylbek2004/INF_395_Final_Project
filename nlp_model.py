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

import sqlalchemy
from sqlalchemy import create_engine
import pandas as pd

# Connection parameters
host = "localhost"
database = "ML"
user = "postgres"
password = "1234"
port = "5432"

connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

engine = create_engine(connection_string)

query = "SELECT * FROM ml_project"

df = pd.read_sql(query, engine)


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

joblib.dump(X_test_text, "X_test.pkl")
joblib.dump(y_test, "y_test.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
