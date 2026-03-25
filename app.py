import streamlit as st
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
from nltk.corpus import stopwords

st.title("Hate Speech Detection App")

st.write("Enter text to analyze:")

# Sample dataset
data = {
    "text": [
        "I hate you",
        "You are amazing",
        "This is stupid",
        "I love this",
        "You are dumb",
        "Great work",
        "Terrible person",
        "Nice effort"
    ],
    "label": ["hate", "normal", "offensive", "normal", "offensive", "normal", "hate", "normal"]
}

df = pd.DataFrame(data)

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

df["text"] = df["text"].apply(clean_text)

# Convert text to numbers
cv = CountVectorizer(stop_words=stopwords.words('english'))
X = cv.fit_transform(df["text"])
y = df["label"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# User input
user_input = st.text_area("Enter your text here")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vector = cv.transform([cleaned])
    prediction = model.predict(vector)
    
    st.subheader("Prediction Result")
    
    if prediction[0] == "hate":
        st.error("Hate Speech")
    elif prediction[0] == "offensive":
        st.warning("Offensive Language")
    else:
        st.success("Normal Text")