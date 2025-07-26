import streamlit as st
import pandas as pd
import string
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Cache model training for performance
@st.cache_data
def train_model():
    # Load and clean dataset
    df = pd.read_csv("spam_data.csv", encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['message'] = df['message'].apply(clean_text)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Split and vectorize
    X_train, _, y_train, _ = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train)

    # Train model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    return model, vectorizer

# Streamlit UI
st.set_page_config(page_title="Spam Classifier", page_icon="📨")
st.title("📨 Spam Message Classifier")
st.write("Enter a message and check if it's **Spam** or **Not Spam (Ham)**.")

# Input field
message = st.text_area("Enter your message here:", height=150)

# Predict button
if st.button("Classify"):
    if not message.strip():
        st.warning("⚠️ Please enter a message.")
    else:
        model, vectorizer = train_model()
        cleaned = clean_text(message)
        vec_msg = vectorizer.transform([cleaned])
        pred = model.predict(vec_msg)[0]

        if pred == 1:
            st.error("🚫 This message is **Spam**.")
        else:
            st.success("✅ This message is **Not Spam (Ham)**.")
