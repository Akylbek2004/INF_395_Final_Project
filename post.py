import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import zipfile
import re  


# Download NLTK stopwords
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
punc = string.punctuation

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

conn = psycopg2.connect(
    host="localhost",    
    database="ML", 
    user="postgres", 
    password="1234",
    port="5432"
)

engine = create_engine('postgresql+psycopg2://postgres:1234@localhost:5432/ML')

df.to_sql('ml_project', engine, if_exists='replace', index=False)

conn.close()
