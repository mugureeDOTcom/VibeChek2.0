# streamlit_app.py
import streamlit as st
import pandas as pd
import re
import time
from serpapi.google_search_results import GoogleSearch  # Correct import
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64
from collections import Counter

# Download NLTK data at startup
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

# Page configuration
st.set_page_config(page_title="VibeScraper AI Dashboard", layout="wide")
st.title("🧠 VibeScraper: Google Review Analyzer")

st.markdown("""
**Don't know your Place ID?**
🔗 [Find your Google Place ID here](https://developers.google.com/maps/documentation/places/web-service/place-id)
Search for your business and copy the Place ID.
""")

# Get API Key from secrets
try:
    SERPAPI_KEY = st.secrets["SERPAPI_KEY"]
except Exception:
    SERPAPI_KEY = st.text_input("Enter your SerpAPI Key", type="password")
    if not SERPAPI_KEY:
        st.warning("Please enter your SerpAPI Key to continue")
        st.stop()

# Initialize session state for storing data
if "reviews_df" not in st.session_state:
    st.session_state.reviews_df = None

# Input Place ID
place_id = st.text_input("📍 Enter Google Place ID")

# Max Reviews
max_reviews = st.slider("🔄 How many reviews to fetch?", min_value=50, max_value=500, step=50, value=150)

if st.button("🚀 Fetch & Analyze Reviews") and place_id:
    # Function to clean text
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        return text.strip().lower()
    
    try:
        with st.spinner("Fetching reviews from Google Maps..."):
            # Create params with error handling
            params = {
                "engine": "google_maps_reviews",
                "place_id": place_id,
                "api_key": SERPAPI_KEY,
            }
            
            # Test API connection with just one request first
            try:
                # Explicitly use the correct import path
                test_search = GoogleSearch(params)
                test_results = test_search.get_dict()
                
                if "error" in test_results:
                    st.error(f"API Error: {test_results['error']}")
                    st.stop()
                    
                if "reviews" not in test_results:
                    st.warning("No reviews found for this Place ID. Please verify the ID is correct.")
                    st.stop()
                    
            except Exception as e:
                st.error(f"Error connecting to SerpAPI: {str(e)}")
                st.stop()
            
            # Now fetch all reviews
            all_reviews = []
            start = 0
            
            progress_bar = st.progress(0)
            
            # Make multiple requests with pagination
            while len(all_reviews) < max_reviews:
                params["start"] = start
                
                try:
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
                    
                    # Sleep to respect API rate limits
                    time.sleep(2)
                    
                except Exception as e:
                    st.warning(f"Error during pagination (fetched {len(all_reviews)} reviews so far): {str(e)}")
                    break
            
            if not all_reviews:
                st.error("No reviews could be fetched. Please check your Place ID and API key.")
                st.stop()
                
            df = pd.DataFrame(all_reviews[:max_reviews])
            
            # Handle missing columns - sometimes SerpAPI response structure varies
            for col in ['snippet', 'rating']:
                if col not in df.columns:
                    df[col] = None
            
            # Basic data validation
            df = df.dropna(subset=['snippet'])
            
            if len(df) == 0:
                st.error("No valid reviews found after filtering.")
                st.stop()
                
            st.session_state.reviews_df = df
            st.success(f"✅ {len(df)} reviews fetched!")
    
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        st.stop()
    
    # Process the data
    try:
        with st.spinner("Processing reviews..."):
            # Clean reviews
            df["Cleaned_Review"] = df["snippet"].apply(clean_text)
            
            # Simple ratings analysis
            if "rating" in df.columns and df["rating"].notna().any():
                fig, ax = plt.subplots()
                rating_counts = df["rating"].value_counts().sort_index()
                ax.bar(rating_counts.index, rating_counts.values)
                ax.set_xlabel("Rating")
                ax.set_ylabel("Count")
                ax.set_title("Rating Distribution")
                st.pyplot(fig)
            
            # Sentiment Analysis with VADER only
            sia = SentimentIntensityAnalyzer()
            
            def vader_sentiment(text):
                if not text:
                    return "Neutral"
                score = sia.polarity_scores(text)["compound"]
                return "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"
            
            df["Sentiment"] = df["Cleaned_Review"].apply(vader_sentiment)
            
            # Show sentiment distribution
            st.subheader("📊 Sentiment Analysis")
            sentiment_counts = df["Sentiment"].value_counts()
            
            fig, ax = plt.subplots()
            colors = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
            ax.pie(
                sentiment_counts, 
                labels=sentiment_counts.index, 
                autopct='%1.1f%%',
                colors=[colors.get(x, 'blue') for x in sentiment_counts.index]
            )
            ax.set_title("Sentiment Distribution")
            st.pyplot(fig)
            
            # Show the data
            st.subheader("📋 Review Data")
            st.dataframe(df[["snippet", "Sentiment"]].head(10))
    
    except Exception as e:
        st.error(f"Error in data processing: {str(e)}")
    
    # Word Clouds - only if we have enough data
    try:
        if len(df) > 5:
            st.subheader("☁️ Word Clouds")
            col1, col2 = st.columns(2)
            
            with col1:
                pos_reviews = df[df["Sentiment"] == "Positive"]["Cleaned_Review"].dropna()
                if len(pos_reviews) > 0:
                    pos_text = " ".join(pos_reviews)
                    if len(pos_text) > 50:  # Ensure we have enough text
                        wc_pos = WordCloud(width=400, height=300, background_color="white").generate(pos_text)
                        fig, ax = plt.subplots()
                        ax.imshow(wc_pos, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                        st.caption("Positive Reviews")
                    else:
                        st.info("Not enough positive review text for word cloud")
                else:
                    st.info("No positive reviews found")
            
            with col2:
                neg_reviews = df[df["Sentiment"] == "Negative"]["Cleaned_Review"].dropna()
                if len(neg_reviews) > 0:
                    neg_text = " ".join(neg_reviews)
                    if len(neg_text) > 50:  # Ensure we have enough text
                        wc_neg = WordCloud(width=400, height=300, background_color="black", colormap="Reds").generate(neg_text)
                        fig, ax = plt.subplots()
                        ax.imshow(wc_neg, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                        st.caption("Negative Reviews")
                    else:
                        st.info("Not enough negative review text for word cloud")
                else:
                    st.info("No negative reviews found")
    
    except Exception as e:
        st.warning(f"Error generating word clouds: {str(e)}")
    
    # Top words analysis - simple word frequency analysis
    try:
        st.subheader("🔍 Common Words Analysis")
        
        def get_top_words(reviews, n=10):
            if not reviews.any():
                return []
                
            # Combine all review text
            all_text = " ".join(reviews)
            
            # Split into words and count
            words = re.findall(r'\b\w+\b', all_text.lower())
            
            # Simple stopwords filtering
            stopwords = {'the', 'a', 'an', 'and', 'is', 'in', 'it', 'to', 'was', 'for', 
                         'of', 'with', 'on', 'at', 'by', 'this', 'that', 'but', 'are', 
                         'be', 'or', 'have', 'has', 'had', 'not', 'what', 'all', 'were', 
                         'when', 'where', 'who', 'which', 'their', 'they', 'them', 'there',
                         'from', 'out', 'some', 'would', 'about', 'been', 'many', 'us', 'we'}
            
            # Filter out stopwords and short words
            filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
            
            # Count word frequency
            word_counts = Counter(filtered_words)
            
            # Return top N words
            return word_counts.most_common(n)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Positive Review Keywords")
            pos_words = get_top_words(df[df["Sentiment"] == "Positive"]["Cleaned_Review"])
            
            if pos_words:
                # Create a bar chart
                pos_df = pd.DataFrame(pos_words, columns=['Word', 'Count'])
                fig, ax = plt.subplots()
                ax.barh(pos_df['Word'][::-1], pos_df['Count'][::-1], color='green')
                ax.set_title("Top Words in Positive Reviews")
                st.pyplot(fig)
            else:
                st.info("Not enough data for positive keyword analysis")
        
        with col2:
            st.markdown("#### Negative Review Keywords")
            neg_words = get_top_words(df[df["Sentiment"] == "Negative"]["Cleaned_Review"])
            
            if neg_words:
                # Create a bar chart
                neg_df = pd.DataFrame(neg_words, columns=['Word', 'Count'])
                fig, ax = plt.subplots()
                ax.barh(neg_df['Word'][::-1], neg_df['Count'][::-1], color='red')
                ax.set_title("Top Words in Negative Reviews")
                st.pyplot(fig)
            else:
                st.info("Not enough data for negative keyword analysis")
    
    except Exception as e:
        st.warning(f"Error in keyword analysis: {str(e)}")
    
    # Download Results
    try:
        st.subheader("📎 Download Your Results")
        
        # Create download button for dataframe
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')
        
        if st.session_state.reviews_df is not None:
            csv = convert_df_to_csv(df)
            st.download_button(
                label="📥 Download Reviews CSV",
                data=csv,
                file_name="review_analysis.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.warning(f"Error with download functionality: {str(e)}")

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
        - 🔍 **Common Words Analysis**: Discover what words customers mention most often
        - ☁️ **Word Clouds**: Visualize common words in positive and negative reviews
        - 📊 **Sentiment Analysis**: AI-powered sentiment detection using VADER
        - 💡 **Key Insights**: Get actionable business insights from your reviews
        """)
