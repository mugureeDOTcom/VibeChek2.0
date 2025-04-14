# streamlit_app.py
import streamlit as st
import pandas as pd
import re
import time
from serpapi import GoogleSearch
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
import os
import base64

nltk.download("vader_lexicon")
nltk.download('stopwords')

st.set_page_config(page_title="VibeScraper AI Dashboard", layout="wide")
st.title("🧠 VibeScraper: Google Review Analyzer")

st.markdown("""
**Don't know your Place ID?**
🔗 [Find your Google Place ID here](https://developers.google.com/maps/documentation/places/web-service/place-id)
Search for your business and copy the Place ID.
""")

# Get API Key
SERPAPI_KEY = st.secrets["SERPAPI_KEY"]

# Input Place ID
place_id = st.text_input("📍 Enter Google Place ID")

# Max Reviews
max_reviews = st.slider("🔄 How many reviews to fetch?", min_value=100, max_value=1000, step=100, value=300)

if st.button("🚀 Fetch & Analyze Reviews") and place_id:
    with st.spinner("Fetching reviews from Google Maps..."):
        all_reviews = []
        start = 0

        while len(all_reviews) < max_reviews:
            params = {
                "engine": "google_maps_reviews",
                "place_id": place_id,
                "api_key": SERPAPI_KEY,
                "start": start
            }
            search = GoogleSearch(params)
            results = search.get_dict()
            reviews = results.get("reviews", [])
            if not reviews:
                break
            all_reviews.extend(reviews)
            start += len(reviews)
            time.sleep(1.5)

        df = pd.DataFrame(all_reviews[:max_reviews])
        df.to_csv("reviews.csv", index=False)
        st.success(f"✅ {len(df)} reviews fetched and saved!")

        # Clean reviews
        def clean_text(text):
            text = re.sub(r"http\S+", "", text)
            text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
            return text.strip().lower()

        df["Cleaned_Review"] = df["snippet"].astype(str).apply(clean_text)
        df.to_csv("cleaned_reviews.csv", index=False)

        # Sentiment Analysis
        sia = SentimentIntensityAnalyzer()
        transformer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

        def vader_sentiment(text):
            score = sia.polarity_scores(text)["compound"]
            return "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"

        def bert_sentiment(text):
            result = transformer(text)[0]["label"]
            return "Positive" if "5" in result or "4" in result else "Negative" if "1" in result or "2" in result else "Neutral"

        df["Sentiment_VADER"] = df["Cleaned_Review"].apply(vader_sentiment)
        df["Sentiment_BERT"] = df["Cleaned_Review"].apply(bert_sentiment)
        df.to_csv("sentiment_results.csv", index=False)

        st.success("✅ Sentiment Analysis Complete!")
        st.dataframe(df[["snippet", "Cleaned_Review", "Sentiment_VADER", "Sentiment_BERT"]].head())

        # Word Clouds
        st.subheader("☁️ Word Clouds")
        col1, col2 = st.columns(2)

        with col1:
            pos_text = " ".join(df[df["Sentiment_VADER"] == "Positive"]["Cleaned_Review"])
            wc_pos = WordCloud(width=400, height=300, background_color="white").generate(pos_text)
            st.image(wc_pos.to_array(), caption="Positive Reviews")

        with col2:
            neg_text = " ".join(df[df["Sentiment_VADER"] == "Negative"]["Cleaned_Review"])
            wc_neg = WordCloud(width=400, height=300, background_color="black").generate(neg_text)
            st.image(wc_neg.to_array(), caption="Negative Reviews")

        # Topic Extraction
        def extract_topics(reviews):
            vectorizer = CountVectorizer(stop_words="english", max_features=1000)
            X = vectorizer.fit_transform(reviews)
            lda = LatentDirichletAllocation(n_components=3, random_state=42)
            lda.fit(X)
            topics = []
            for topic in lda.components_:
                words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
                topics.append(", ".join(words))
            return topics

        neg_topics = extract_topics(df[df["Sentiment_VADER"] == "Negative"]["Cleaned_Review"])
        pos_topics = extract_topics(df[df["Sentiment_VADER"] == "Positive"]["Cleaned_Review"])

        # AI Recommendations
        st.subheader("🤖 AI Recommendations")
        gen = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

        with open("recommendations.txt", "w") as file:
            file.write("📌 Business Recommendations Based on Customer Feedback\n\n")
            file.write("🔴 Areas for Improvement:\n")
            for issue in neg_topics:
                prompt = f"Customers have complained about {issue}. What are the best ways a business can improve this?"
                rec = gen(prompt, max_length=100, do_sample=True)[0]["generated_text"]
                file.write(f"- {rec}\n\n")
                st.markdown(f"**Improvement:** {rec}")

            file.write("🟢 Strengths to Maintain:\n")
            for pos in pos_topics:
                prompt = f"Customers love {pos}. How can a business continue excelling in this area?"
                rec = gen(prompt, max_length=100, do_sample=True)[0]["generated_text"]
                file.write(f"- {rec}\n\n")
                st.markdown(f"**Strength Insight:** {rec}")

        # Download Buttons
        def download_file(filename, label):
            with open(filename, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">📥 Download {label}</a>'
                st.markdown(href, unsafe_allow_html=True)

        st.markdown("### 📎 Download Your Results")
        download_file("reviews.csv", "Raw Reviews CSV")
        download_file("cleaned_reviews.csv", "Cleaned Reviews CSV")
        download_file("sentiment_results.csv", "Sentiment CSV")
        download_file("recommendations.txt", "AI Recommendations")
