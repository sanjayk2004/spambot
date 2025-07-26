import streamlit as st
import pickle
import re
import string

# Load model and vectorizer
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Streamlit App UI
st.title("📩 Spam Detection System")
st.markdown("Using **NLP + Machine Learning** to classify messages as Spam or Ham")

# Input from user
user_input = st.text_area("Enter a message to check if it's spam or not:")

if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        cleaned = clean_text(user_input)
        vect_text = vectorizer.transform([cleaned])
        prediction = model.predict(vect_text)[0]
        label = "📬 Ham (Not Spam)" if prediction == 0 else "🚫 Spam"
        st.success(f"Prediction: **{label}**")
