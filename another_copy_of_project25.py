# streamlit_app.py
import streamlit as st
import pandas as pd
import re
import time
# Try to import GoogleSearch from the original package, otherwise use our custom implementation
try:
    from google_search_results import GoogleSearch
except ImportError:
    try:
        from serpapi import GoogleSearch
    except ImportError:
        st.warning("SerpAPI package not found. Using custom implementation.")
        # Import our custom implementation
        from serpapi_wrapper import GoogleSearch

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
st.set_page_config(page_title="VibeChek AI Dashboard", layout="wide")
st.title("🧠 VibeChek: Google Review Analyzer")

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

# Add sorting options
sort_option = st.selectbox(
    "🔄 Sort reviews by:",
    options=["Most Recent First", "Highest Rating First", "Lowest Rating First"],
    index=0
)

# Add date filtering options
st.subheader("Advanced Options")
use_date_filter = st.checkbox("Filter by date range", value=False)

if use_date_filter:
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=None)
    with col2:
        end_date = st.date_input("End date", value=None)

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
            
            # Add sorting parameter based on user selection
            if sort_option == "Most Recent First":
                params["sort"] = "newest"  # This is actually the default
            elif sort_option == "Highest Rating First":
                params["sort"] = "highest_rating"
            elif sort_option == "Lowest Rating First":
                params["sort"] = "lowest_rating"
            
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
            
            # Improved status display
            status_container = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Make multiple requests with pagination
            while len(all_reviews) < max_reviews:
                params["start"] = start
                
                try:
                    # Update status before each API call
                    status_container.info(f"Fetching reviews (batch {start//10 + 1})...")
                    
                    search = GoogleSearch(params)
                    results = search.get_dict()
                    reviews = results.get("reviews", [])
                    
                    if not reviews:
                        status_container.warning("No more reviews available")
                        break
                        
                    all_reviews.extend(reviews)
                    start += len(reviews)
                    
                    # Update progress and status text
                    progress = min(len(all_reviews) / max_reviews, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Fetched {len(all_reviews)}/{max_reviews} reviews")
                    
                    # Sleep to respect API rate limits
                    time.sleep(2)
                    
                except Exception as e:
                    status_container.warning(f"Error during pagination (fetched {len(all_reviews)} reviews so far): {str(e)}")
                    break
            
            # Update final status
            if len(all_reviews) > 0:
                status_container.success(f"✅ Successfully fetched {len(all_reviews)} reviews!")
            else:
                status_container.error("Failed to fetch any reviews")
                st.stop()
                
            df = pd.DataFrame(all_reviews[:max_reviews])
            
            # Handle missing columns - sometimes SerpAPI response structure varies
            for col in ['snippet', 'rating', 'time']:
                if col not in df.columns:
                    df[col] = None
            
            # Basic data validation
            df = df.dropna(subset=['snippet'])
            
            if len(df) == 0:
                st.error("No valid reviews found after filtering.")
                st.stop()
            
            # Date filtering
            if use_date_filter and "time" in df.columns and df["time"].notna().any():
                # Convert timestamp to datetime
                df["date"] = pd.to_datetime(df["time"].astype(float), unit='s', errors='coerce')
                
                # Apply date filters
                if start_date:
                    df = df[df["date"] >= pd.Timestamp(start_date)]
                if end_date:
                    df = df[df["date"] <= pd.Timestamp(end_date)]
                
                if len(df) == 0:
                    st.warning("No reviews found in the selected date range")
                    st.stop()
                else:
                    st.success(f"Filtered to {len(df)} reviews in the selected date range")
                
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
            
            # Updated ratings analysis code
            if "rating" in df.columns and df["rating"].notna().any():
                try:
                    # Convert to numeric, coerce errors to NaN
                    df["rating_numeric"] = pd.to_numeric(df["rating"], errors="coerce")
                    
                    # Drop NaN values and ensure we still have data
                    df_ratings = df.dropna(subset=["rating_numeric"])
                    
                    if len(df_ratings) > 0:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        
                        # Round to nearest integer if needed (sometimes ratings might be like 4.7)
                        df_ratings["rating_rounded"] = df_ratings["rating_numeric"].round().astype(int)
                        
                        # Get counts and sort
                        rating_counts = df_ratings["rating_rounded"].value_counts().sort_index()
                        
                        # Create a bar chart with clear labels
                        bars = ax.bar(rating_counts.index, rating_counts.values, color="skyblue")
                        
                        # Add values on top of bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                    f'{int(height)}', ha='center', va='bottom')
                        
                        # Improve x-axis to show all possible ratings (1-5)
                        ax.set_xticks(range(1, 6))
                        ax.set_xlabel("Rating (stars)")
                        ax.set_ylabel("Number of Reviews")
                        ax.set_title("Rating Distribution")
                        st.pyplot(fig)
                    else:
                        st.info("No valid numerical ratings found in the data.")
                except Exception as e:
                    st.warning(f"Error generating ratings chart: {str(e)}")
            else:
                st.info("No rating data available for visualization.")
            
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
            
            fig, ax = plt.subplots(figsize=(5, 4))
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
    
    # Timeline Analysis - trends over time
    try:
        if "time" in df.columns and df["time"].notna().any():
            st.subheader("📈 Sentiment Timeline")
            
            # Convert time values to datetime (SerpAPI returns timestamps)
            df["date"] = pd.to_datetime(df["time"].astype(float), unit='s', errors='coerce')
            df = df.dropna(subset=["date"])  # Remove rows with invalid dates
            
            if len(df) > 0:
                df["month_year"] = df["date"].dt.strftime('%Y-%m')
                
                # Group by month and sentiment
                timeline_data = df.groupby(["month_year", "Sentiment"]).size().reset_index(name="count")
                
                # Pivot to get sentiment as columns
                pivot_data = timeline_data.pivot(index="month_year", columns="Sentiment", values="count").reset_index()
                pivot_data = pivot_data.fillna(0)
                
                # Make sure all sentiment types exist as columns
                for sentiment in ["Positive", "Negative", "Neutral"]:
                    if sentiment not in pivot_data.columns:
                        pivot_data[sentiment] = 0
                
                # Sort by date
                pivot_data["month_year"] = pd.to_datetime(pivot_data["month_year"], format='%Y-%m')
                pivot_data = pivot_data.sort_values("month_year")
                pivot_data["month_year"] = pivot_data["month_year"].dt.strftime('%Y-%m')
                
                # Create the timeline visualization with smaller size
                fig, ax = plt.subplots(figsize=(8, 4))
                
                # Plot each sentiment type
                ax.plot(pivot_data["month_year"], pivot_data["Positive"], marker='o', color='green', label='Positive')
                ax.plot(pivot_data["month_year"], pivot_data["Negative"], marker='x', color='red', label='Negative')
                if "Neutral" in pivot_data.columns:
                    ax.plot(pivot_data["month_year"], pivot_data["Neutral"], marker='s', color='gray', label='Neutral')
                
                # Rotate x-axis labels for better readability
                plt.xticks(rotation=45, ha='right')
                
                # Add labels and legend
                ax.set_xlabel("Month")
                ax.set_ylabel("Number of Reviews")
                ax.set_title("Sentiment Trends Over Time")
                ax.legend()
                
                # Improve layout
                plt.tight_layout()
                
                # Show the plot
                st.pyplot(fig)
                
                # Show insights about the timeline
                if len(pivot_data) > 1:
                    latest_month = pivot_data.iloc[-1]
                    previous_month = pivot_data.iloc[-2] if len(pivot_data) > 1 else None
                    
                    if previous_month is not None:
                        pos_change = latest_month["Positive"] - previous_month["Positive"]
                        neg_change = latest_month["Negative"] - previous_month["Negative"]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Recent Positive Trend:**")
                            if pos_change > 0:
                                st.success(f"✅ Positive reviews increased by {int(pos_change)} in the latest month")
                            elif pos_change < 0:
                                st.warning(f"⚠️ Positive reviews decreased by {abs(int(pos_change))} in the latest month")
                            else:
                                st.info("Positive reviews remained stable in the latest month")
                                
                        with col2:
                            st.markdown("**Recent Negative Trend:**")
                            if neg_change > 0:
                                st.error(f"❌ Negative reviews increased by {int(neg_change)} in the latest month")
                            elif neg_change < 0:
                                st.success(f"✅ Negative reviews decreased by {abs(int(neg_change))} in the latest month")
                            else:
                                st.info("Negative reviews remained stable in the latest month")
            else:
                st.info("Not enough date information for timeline analysis")
        else:
            st.info("Timeline analysis not available - review data doesn't include timestamps")
    except Exception as e:
        st.warning(f"Error generating timeline: {str(e)}")
    
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
                        fig, ax = plt.subplots(figsize=(4, 3))
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
                        fig, ax = plt.subplots(figsize=(4, 3))
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
                # Create a bar chart with smaller size
                pos_df = pd.DataFrame(pos_words, columns=['Word', 'Count'])
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.barh(pos_df['Word'][::-1], pos_df['Count'][::-1], color='green')
                ax.set_title("Top Words in Positive Reviews")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Not enough data for positive keyword analysis")
        
        with col2:
            st.markdown("#### Negative Review Keywords")
            neg_words = get_top_words(df[df["Sentiment"] == "Negative"]["Cleaned_Review"])
            
            if neg_words:
                # Create a bar chart with smaller size
                neg_df = pd.DataFrame(neg_words, columns=['Word', 'Count'])
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.barh(neg_df['Word'][::-1], neg_df['Count'][::-1], color='red')
                ax.set_title("Top Words in Negative Reviews")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Not enough data for negative keyword analysis")
    
    except Exception as e:
        st.warning(f"Error in keyword analysis: {str(e)}")
    
    # Simple AI Recommendations
    try:
        st.subheader("🤖 Smart Recommendations")
        
        # Define common business issues and solutions based on keywords
        business_insights = {
            "service": {
                "positive": "Your service is praised by customers. Continue training staff on excellent customer service techniques.",
                "negative": "Service issues appear in negative reviews. Consider staff training or reviewing service protocols."
            },
            "price": {
                "positive": "Customers find your pricing reasonable and fair. Maintain this pricing strategy.",
                "negative": "Price concerns appear in negative reviews. Consider reviewing your pricing strategy or better communicating value."
            },
            "quality": {
                "positive": "Product/service quality is appreciated. Maintain your quality standards.",
                "negative": "Quality concerns appear in reviews. Review quality control processes."
            },
            "wait": {
                "positive": "Customers appreciate your efficient timing/waiting periods.",
                "negative": "Wait times appear to be an issue. Consider operational efficiency improvements."
            },
            "clean": {
                "positive": "Cleanliness is noted positively. Maintain your cleanliness standards.",
                "negative": "Cleanliness concerns appear in reviews. Review cleaning protocols."
            },
            "staff": {
                "positive": "Your staff receives positive mentions. Recognize and reward good employees.",
                "negative": "Staff-related concerns appear in reviews. Consider additional training or reviewing hiring practices."
            },
            "location": {
                "positive": "Your location is mentioned positively. Highlight this in marketing materials.",
                "negative": "Location issues appear in reviews. Consider improving signage, access, or parking if possible."
            },
            "food": {
                "positive": "Food quality is praised. Maintain your food preparation standards.",
                "negative": "Food quality issues appear in reviews. Review kitchen operations and quality control."
            },
            "atmosphere": {
                "positive": "Customers enjoy your atmosphere/ambiance. Maintain this environment.",
                "negative": "Atmosphere concerns appear in reviews. Consider refreshing your space's design or ambiance."
            },
            "parking": {
                "positive": "Parking is mentioned positively. Continue to maintain good parking options.",
                "negative": "Parking issues appear in reviews. Consider improving parking options or providing clearer instructions."
            },
            "menu": {
                "positive": "Your menu receives positive attention. Continue with your current menu strategy.",
                "negative": "Menu concerns appear in reviews. Consider updating options or improving descriptions."
            },
            "value": {
                "positive": "Customers find good value in your offerings. Maintain this balance of price and quality.",
                "negative": "Value concerns appear in reviews. Review pricing or improve quality to increase perceived value."
            },
            "friendly": {
                "positive": "Friendliness is mentioned positively. Continue encouraging friendly customer interactions.",
                "negative": "Consider emphasizing a more friendly approach to customer service."
            },
            "recommend": {
                "positive": "Customers are recommending your business - excellent! Consider a referral program.",
                "negative": "Work on issues that prevent customers from recommending your business."
            },
            "management": {
                "positive": "Management is mentioned positively. Maintain these management practices.",
                "negative": "Management concerns appear in reviews. Consider reviewing management training or approaches."
            },
            "reservation": {
                "positive": "Your reservation system works well for customers.",
                "negative": "Reservation issues appear in reviews. Review your reservation system or processes."
            },
            "bathroom": {
                "positive": "Restrooms are mentioned positively. Maintain cleanliness standards.",
                "negative": "Bathroom concerns appear in reviews. Improve cleaning frequency or facilities."
            },
            "noisy": {
                "positive": "Customers appreciate the sound levels in your establishment.",
                "negative": "Noise issues appear in reviews. Consider acoustic improvements or music volume adjustments."
            },
            "portion": {
                "positive": "Portion sizes are mentioned positively. Maintain current portion standards.",
                "negative": "Portion size concerns appear in reviews. Review your portion sizing strategy."
            },
            "delivery": {
                "positive": "Delivery service is praised by customers. Maintain delivery standards.",
                "negative": "Delivery issues appear in reviews. Review delivery processes and timing."
            },
            "return": {
                "positive": "Customers mention returning to your business - excellent repeat business!",
                "negative": "Consider addressing issues that prevent customers from returning."
            }
        }
        
        # Check for keywords in positive and negative reviews
        positive_insights = []
        negative_insights = []
        
        positive_text = " ".join(df[df["Sentiment"] == "Positive"]["Cleaned_Review"]).lower()
        negative_text = " ".join(df[df["Sentiment"] == "Negative"]["Cleaned_Review"]).lower()
        
        for keyword, insights in business_insights.items():
            if keyword in positive_text:
                positive_insights.append(insights["positive"])
            if keyword in negative_text:
                negative_insights.append(insights["negative"])
        
        # Add insights based on overall sentiment
        if len(df) > 0:
            sentiment_counts = df["Sentiment"].value_counts()
            positive_pct = sentiment_counts.get("Positive", 0) / len(df) * 100 if len(df) > 0 else 0
            negative_pct = sentiment_counts.get("Negative", 0) / len(df) * 100 if len(df) > 0 else 0
            
            if positive_pct > 75:
                positive_insights.append("Overall sentiment is very positive. Consider using positive reviews in marketing materials.")
            elif positive_pct < 50:
                negative_insights.append("Overall sentiment is concerning. Consider a comprehensive review of business operations.")
        
        # Add insights based on most common words
        pos_words = get_top_words(df[df["Sentiment"] == "Positive"]["Cleaned_Review"], 5)
        neg_words = get_top_words(df[df["Sentiment"] == "Negative"]["Cleaned_Review"], 5)
        
        if pos_words:
            top_pos_words = [word for word, count in pos_words]
            positive_insights.append(f"Customers often mention '{', '.join(top_pos_words)}' positively. Emphasize these aspects in marketing.")
            
        if neg_words:
            top_neg_words = [word for word, count in neg_words]
            negative_insights.append(f"Customers often mention '{', '.join(top_neg_words)}' negatively. Address these areas for improvement.")
        
        # Display recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🟢 Strengths to Maintain")
            if positive_insights:
                for i, insight in enumerate(positive_insights[:5], 1):  # Limit to top 5
                    st.markdown(f"{i}. {insight}")
            else:
                st.info("Not enough data to generate strength recommendations")
        
        with col2:
            st.markdown("### 🔴 Areas for Improvement")
            if negative_insights:
                for i, insight in enumerate(negative_insights[:5], 1):  # Limit to top 5
                    st.markdown(f"{i}. {insight}")
            else:
                st.info("Not enough data to generate improvement recommendations")
                
        # Add a section for actionable next steps
        if positive_insights or negative_insights:
            st.markdown("### 📝 Actionable Next Steps")
            
            action_items = []
            
            if negative_insights:
                # Prioritize addressing top negative issues
                action_items.append("Address the top negative themes in customer reviews with specific improvement plans")
                
            if positive_insights:
                # Leverage strengths
                action_items.append("Highlight positive aspects in marketing materials and train staff to emphasize these strengths")
            
            # Add general best practices
            action_items.extend([
                "Respond to negative reviews promptly and professionally",
                "Track sentiment trends monthly to measure improvement",
                "Implement a customer feedback system to catch issues before they result in negative reviews",
                "Train staff on common customer pain points identified in the analysis"
            ])
            
            for i, action in enumerate(action_items,for i, action in enumerate(action_items, 1):
                st.markdown(f"{i}. {action}")
            
    except Exception as e:
        st.warning(f"Error generating recommendations: {str(e)}")
    
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
        - 📈 **Sentiment Timeline**: Track how sentiment changes over time
        - 🔍 **Common Words Analysis**: Discover what words customers mention most often
        - ☁️ **Word Clouds**: Visualize common words in positive and negative reviews
        - 📊 **Sentiment Analysis**: AI-powered sentiment detection using VADER
        - 💡 **Smart Recommendations**: Get actionable business advice based on customer feedback
        """)
