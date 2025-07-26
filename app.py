import streamlit as st
import pandas as pd
import string
import re
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load dataset from URL
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    response = requests.get(url)
    data = StringIO(response.text)
    df = pd.read_csv(data, sep='\t', names=["label", "message"])
    return df

# Train model
@st.cache_data
def train_model(df):
    df['message'] = df['message'].apply(clean_text)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    X_train, _, y_train, _ = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    return model, vectorizer

# Streamlit UI
st.set_page_config(page_title="Spam Classifier", page_icon="📨")
st.title("📨 Spam Message Classifier (Auto Dataset)")
st.write("Enter a message and check if it's **Spam** or **Ham (Not Spam)**.")

# Load and train
with st.spinner("Loading model..."):
    df = load_data()
    model, vectorizer = train_model(df)

# User input
user_input = st.text_area("Enter your message here:", height=150)

# Predict
if st.button("Classify"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a message.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        if prediction == 1:
            st.error("🚫 This message is **Spam**.")
        else:
            st.success("✅ This message is **Not Spam (Ham)**.")
