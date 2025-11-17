# app.py
import streamlit as st
import joblib

def dummy(x):
    return x

@st.cache_resource
def load_models():
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    model = joblib.load('calibrated_clf.joblib')
    encoder = joblib.load('encoder.joblib')
    return vectorizer, model, encoder

vectorizer, model, encoder = load_models()

st.title("Hand Gesture Classification")
st.write("Enter any text and click Predict")

text = st.text_area("Text:", height=150)

if st.button("Predict"):
    if text.strip():
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        label = encoder.inverse_transform([pred])[0]
        st.success(f"Prediction: **{label}**")
    else:
        st.error("Please enter some text first!")