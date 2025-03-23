import streamlit as st 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch 
import torch.nn.functional as F
import pandas as pd

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained("berkutayozdemir/yelp-sentiment-star-prediction")
    model = AutoModelForSequenceClassification.from_pretrained("berkutayozdemir/yelp-sentiment-star-prediction")
    return tokenizer, model

tokenizer, model = load_model(model_name="yelp_sentiment_multiclass_model")

st.title("Yelp Review Star Predictor With Transformers")
st.write("Enter a review below and get a star rating prediction:")

mode = st.radio("Select Output Mode:", ["⭐ Star Rating", "😊 Sentiment Label"])

text = st.text_area("Enter Your Review : ")

if st.button("Analyze"):
    inputs = tokenizer(text, return_tensors = "pt", padding = True, truncation = True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim = 1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    if mode == "⭐ Star Rating":
        st.success(f"⭐ **Predicted Rating:** {pred_class + 1} stars (Confidence: {confidence:.2f})")
        probs_df = pd.DataFrame(probs.numpy()[0], index=[1, 2, 3, 4, 5], columns=["Probability"])
        st.bar_chart(probs_df)

    elif mode == "😊 Sentiment Label":
        # Convert star rating to sentiment
        if pred_class <= 1:
            sentiment = "Negative 😡"
        elif pred_class == 2:
            sentiment = "Neutral 😐"
        else:
            sentiment = "Positive 😄"
        st.success(f"**Sentiment:** {sentiment} (Confidence: {confidence:.2f})")

        