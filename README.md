# Yelp Review Star Predictor & Sentiment App

A **fine-grained sentiment-analysis project** that predicts the **exact 1–5 star rating** of a Yelp review —and can also output a simple **Positive / Neutral / Negative** label with a click.

| Demo | Model | Dataset |
|------|-------|---------|
| **[Live Streamlit App](https://sentiment-analysis-yelp.streamlit.app)** | **[`berkutayozdemir/yelp-sentiment-star-prediction`](https://huggingface.co/berkutayozdemir/yelp-sentiment-star-prediction)** | [`yelp_review_full`](https://huggingface.co/datasets/yelp_review_full) |

---

## ✨ Features
- **5-class rating** (1⭐ – 5⭐) from plain text
- **Sentiment toggle**: full stars **or** Positive / Neutral / Negative
- **Probability bar chart** for model confidence
- **DistilBERT** fine-tuned on 650k Yelp reviews
- Fully serverless: model weights are pulled from the **🤗 Hub**, so no giant files in the repo
- One-click deploy to **Streamlit Cloud**

---

## Project Structure
```
sentiment-analysis-yelp/
├── sent_app.py                        # Streamlit UI
├── requirements.txt                   # Slim, cloud-friendly deps
├── .gitignore
├── yelp_sentiment_multiclass_tokenizer/
│   ├── tokenizer.json
│   └── …                              # Other tokenizer files
└── yelp_sentiment_multiclass_model/
    └── config.json                   # Model config (tiny)
```



