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

# Download NLTK data at startup to avoid runtime errors
try:
    nltk.data.find('vader_lexicon')
    nltk.data.find('stopwords')
except LookupError:
    nltk.download("vader_lexicon")
    nltk.download('stopwords')

# Page configuration
st.set_page_config(page_title="VibeScraper AI Dashboard", layout="wide")
st.title("🧠 VibeScraper: Google Review Analyzer")

st.markdown("""
**Don't know your Place ID?**
🔗 [Find your Google Place ID here](https://developers.google.com/maps/documentation/places/web-service/place-id)
Search for your business and copy the Place ID.
""")

# Get API Key
try:
    SERPAPI_KEY = st.secrets["SERPAPI_KEY"]
except Exception:
    st.error("❌ API key not found in secrets. Please add your SERPAPI_KEY to the app secrets.")
    st.stop()

# Initialize session state for storing data
if "reviews_df" not in st.session_state:
    st.session_state.reviews_df = None
if "sentiment_df" not in st.session_state:
    st.session_state.sentiment_df = None

# Input Place ID
place_id = st.text_input("📍 Enter Google Place ID")

# Max Reviews
max_reviews = st.slider("🔄 How many reviews to fetch?", min_value=100, max_value=1000, step=100, value=300)

if st.button("🚀 Fetch & Analyze Reviews") and place_id:
    # Function to clean text
    def clean_text(text):
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text.strip().lower()
    
    with st.spinner("Fetching reviews from Google Maps..."):
        try:
            all_reviews = []
            start = 0

            progress_bar = st.progress(0)
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
                
                # Update progress
                progress = min(len(all_reviews) / max_reviews, 1.0)
                progress_bar.progress(progress)
                
                time.sleep(1.5)  # Respect API rate limits

            df = pd.DataFrame(all_reviews[:max_reviews])
            st.session_state.reviews_df = df
            st.success(f"✅ {len(df)} reviews fetched!")
            
            # Clean reviews
            df["Cleaned_Review"] = df["snippet"].astype(str).apply(clean_text)
        
        except Exception as e:
            st.error(f"Error fetching reviews: {e}")
            st.stop()
    
    with st.spinner("Performing sentiment analysis..."):
        try:
            # Sentiment Analysis with VADER
            sia = SentimentIntensityAnalyzer()
            
            def vader_sentiment(text):
                score = sia.polarity_scores(text)["compound"]
                return "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"
            
            df["Sentiment_VADER"] = df["Cleaned_Review"].apply(vader_sentiment)
            
            # Load a smaller model for sentiment analysis - instead of loading two separate models
            sentiment_model = "distilbert-base-uncased-finetuned-sst-2-english"
            transformer = pipeline("sentiment-analysis", model=sentiment_model)
            
            # Batch processing to speed up inference
            batch_size = 8
            all_reviews = df["Cleaned_Review"].tolist()
            all_results = []
            
            for i in range(0, len(all_reviews), batch_size):
                batch = all_reviews[i:i+batch_size]
                results = transformer(batch)
                all_results.extend(results)
            
            df["Sentiment_BERT"] = [r["label"] for r in all_results]
            st.session_state.sentiment_df = df
            
            st.success("✅ Sentiment Analysis Complete!")
            st.dataframe(df[["snippet", "Cleaned_Review", "Sentiment_VADER", "Sentiment_BERT"]].head())
        
        except Exception as e:
            st.error(f"Error in sentiment analysis: {e}")
    
    # Only proceed if we have data
    if st.session_state.sentiment_df is not None:
        df = st.session_state.sentiment_df
        
        # Analysis Section
        st.header("📊 Analysis Results")
        
        # Sentiment Distribution
        st.subheader("Sentiment Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            vader_counts = df["Sentiment_VADER"].value_counts()
            fig, ax = plt.subplots()
            ax.pie(vader_counts, labels=vader_counts.index, autopct='%1.1f%%')
            ax.set_title("VADER Sentiment")
            st.pyplot(fig)
            
        with col2:
            bert_counts = df["Sentiment_BERT"].value_counts()
            fig, ax = plt.subplots()
            ax.pie(bert_counts, labels=bert_counts.index, autopct='%1.1f%%')
            ax.set_title("BERT Sentiment")
            st.pyplot(fig)
        
        # Word Clouds
        st.subheader("☁️ Word Clouds")
        col1, col2 = st.columns(2)

        with col1:
            pos_text = " ".join(df[df["Sentiment_VADER"] == "Positive"]["Cleaned_Review"])
            if pos_text.strip():
                wc_pos = WordCloud(width=400, height=300, background_color="white").generate(pos_text)
                fig, ax = plt.subplots()
                ax.imshow(wc_pos, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
                st.caption("Positive Reviews")
            else:
                st.info("Not enough positive reviews for word cloud")

        with col2:
            neg_text = " ".join(df[df["Sentiment_VADER"] == "Negative"]["Cleaned_Review"])
            if neg_text.strip():
                wc_neg = WordCloud(width=400, height=300, background_color="black", colormap="Reds").generate(neg_text)
                fig, ax = plt.subplots()
                ax.imshow(wc_neg, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
                st.caption("Negative Reviews")
            else:
                st.info("Not enough negative reviews for word cloud")

        # Topic Extraction
        st.subheader("🔍 Key Topics")
        
        def extract_topics(reviews, n_topics=3):
            if len(reviews) < 10:  # Need minimum data
                return ["Not enough reviews for topic extraction"]
                
            try:
                vectorizer = CountVectorizer(stop_words="english", max_features=500)
                X = vectorizer.fit_transform(reviews)
                
                if X.shape[1] < 10:  # Need minimum features
                    return ["Not enough unique words for topic extraction"]
                    
                lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
                lda.fit(X)
                
                topics = []
                for topic in lda.components_:
                    feature_names = vectorizer.get_feature_names_out()
                    words = [feature_names[i] for i in topic.argsort()[-5:]]
                    topics.append(", ".join(words))
                return topics
            except Exception as e:
                return [f"Error in topic extraction: {str(e)}"]
        
        pos_reviews = df[df["Sentiment_VADER"] == "Positive"]["Cleaned_Review"].tolist()
        neg_reviews = df[df["Sentiment_VADER"] == "Negative"]["Cleaned_Review"].tolist()
        
        pos_topics = extract_topics(pos_reviews)
        neg_topics = extract_topics(neg_reviews)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🟢 Positive Topics")
            for i, topic in enumerate(pos_topics, 1):
                st.markdown(f"**Topic {i}:** {topic}")
                
        with col2:
            st.subheader("🔴 Negative Topics")
            for i, topic in enumerate(neg_topics, 1):
                st.markdown(f"**Topic {i}:** {topic}")
        
        # Recommendations (using a smaller, faster model)
        st.subheader("🤖 AI Recommendations")
        
        # We'll use a pre-trained summarization model instead of text generation
        try:
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            
            # Format recommendations based on topics
            pos_prompt = "Customers love the following aspects: " + ". ".join(pos_topics)
            neg_prompt = "Customers have issues with the following aspects: " + ". ".join(neg_topics)
            
            with st.spinner("Generating recommendations..."):
                # Generate recommendations
                if len(pos_prompt) > 50:
                    pos_summary = summarizer(pos_prompt, max_length=100, min_length=30)[0]["summary_text"]
                    st.markdown("### 🟢 Strengths to Maintain")
                    st.write(pos_summary)
                
                if len(neg_prompt) > 50:
                    neg_summary = summarizer(neg_prompt, max_length=100, min_length=30)[0]["summary_text"]
                    st.markdown("### 🔴 Areas for Improvement")
                    st.write(neg_summary)
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
        
        # Download Results
        st.subheader("📎 Download Your Results")
        
        # Create download buttons for dataframes
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')
        
        if st.session_state.reviews_df is not None:
            csv = convert_df_to_csv(st.session_state.reviews_df)
            st.download_button(
                label="📥 Download Raw Reviews CSV",
                data=csv,
                file_name="reviews.csv",
                mime="text/csv"
            )
        
        if st.session_state.sentiment_df is not None:
            csv = convert_df_to_csv(st.session_state.sentiment_df)
            st.download_button(
                label="📥 Download Sentiment Analysis CSV",
                data=csv,
                file_name="sentiment_results.csv",
                mime="text/csv"
            )
else:
    # Show a placeholder or example when no data is loaded
    if place_id:
        st.info("Click 'Fetch & Analyze Reviews' to start the analysis.")
    else:
        st.info("Enter a Google Place ID and click 'Fetch & Analyze Reviews' to start.")
        
        # Show an example of what the app does
        st.subheader("📱 App Features")
        st.markdown("""
        - 🤖 **Automated Review Analysis**: Get insights from hundreds of reviews in seconds
        - 🔍 **Topic Extraction**: Discover what topics customers mention most often
        - ☁️ **Word Clouds**: Visualize common words in positive and negative reviews
        - 📊 **Sentiment Analysis**: Two advanced AI models analyze customer sentiment
        - 📝 **AI Recommendations**: Get actionable business advice based on customer feedback
        """)
