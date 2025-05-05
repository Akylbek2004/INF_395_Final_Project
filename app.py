import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
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

# Load assets
model = joblib.load("rf_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")
y_test_labels = label_encoder.inverse_transform(y_test)
X_test_vectorized = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vectorized)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Streamlit page settings
st.set_page_config(page_title="Sentiment Classifier", page_icon="🎬")

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

# CSS
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

# Sidebar navigation
page = st.sidebar.selectbox("Select a page", ["🔮 Predict Sentiment", "📊 Model Evaluation", "📘 About"])

# Page 1: Prediction
if page == "🔮 Predict Sentiment":
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

            st.session_state.history.append((user_input, sentiment))

    if st.session_state.history:
        st.subheader("🕓 Prediction History")
        for i, (review, sent) in enumerate(reversed(st.session_state.history[-5:]), 1):
            st.markdown(f"{i}. **{sent}** — _{review[:60]}..._")

# Page 2: Evaluation
elif page == "📊 Model Evaluation":
    st.markdown('<h1 style="text-align: center;">📊 Model Evaluation</h1>', unsafe_allow_html=True)

    st.subheader("Accuracy")
    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Accuracy:** {acc:.4f}")

    st.subheader("Classification Report")
    report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test_labels, y_pred_labels)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)

# Page 3: About
elif page == "📘 About":
    st.markdown('<h1 style="text-align: center;">📘 About This Project</h1>', unsafe_allow_html=True)
    st.write("""
    This project is a **movie review sentiment classifier** built using a `RandomForestClassifier`.
    It uses **TF-IDF** for text vectorization and was trained on labeled review data.

    - Model: Random Forest
    - Vectorizer: TF-IDF
    - Evaluation: Accuracy, Precision, Recall, F1-Score
    - Framework: Streamlit

    You can explore model performance under the **Model Evaluation** tab and try out predictions yourself!
    """)
