
'''1. Sentiment Labeling
Objective

Automatically label each employee message as Positive, Negative, or Neutral.

Approach

We use a Large Language Model (LLM)–based sentiment classifier (can be replaced by VADER/TextBlob if required).
Each message is classified independently, and the result is stored in a new column sentiment_label.

Sentiment Criteria

Positive: Expresses satisfaction, appreciation, motivation.

Negative: Expresses frustration, dissatisfaction, stress.

Neutral: Informational or emotionally balanced messages.

Code  '''
import pandas as pd
from transformers import pipeline

# Load data
df = pd.read_csv("test.csv")

# Sentiment model
sentiment_model = pipeline("sentiment-analysis")

def label_sentiment(text):
    result = sentiment_model(text[:512])[0]['label']
    if result == "POSITIVE":
        return "Positive"
    elif result == "NEGATIVE":
        return "Negative"
    else:
        return "Neutral"

df["sentiment_label"] = df["message"].apply(label_sentiment)
