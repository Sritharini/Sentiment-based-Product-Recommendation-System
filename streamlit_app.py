import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ---------------------------
# SETUP: Load Models and Data
# ---------------------------
tf_idf_vectorizer = pickle.load(open("models/tf_idf_vectorizer.pkl", "rb"))
sentiment_model = pickle.load(open("models/sentiment_model.pkl", "rb"))
similarity_matrix = pickle.load(open("models/similarity_matrix.pkl", "rb"))
review_df = pd.read_csv("data/review_df.csv")

# ---------------------------
# HELPER FUNCTION: Clean Text
# ---------------------------
def clean_text(text):
    return text.lower()

# ---------------------------
# CORE FUNCTION: Hybrid Recommendation
# ---------------------------
def recommend_products(user_id, top_n=20, final_n=5):
    user_item_matrix = pd.pivot_table(review_df, index='user_id', columns='name', values='reviews_rating')

    if user_id not in user_item_matrix.index:
        return pd.DataFrame()

    user_ratings = user_item_matrix.loc[user_id].dropna()
    scores = pd.Series(dtype=float)

    for item, rating in user_ratings.items():
        if item in similarity_matrix:
            similar_items = similarity_matrix[item]
            scores = scores.add(similar_items * rating, fill_value=0)

    scores = scores.drop(user_ratings.index, errors='ignore')
    top_20 = scores.sort_values(ascending=False).head(top_n).reset_index()
    top_20.columns = ['product', 'score']

    filtered_reviews = review_df[review_df['name'].isin(top_20['product'])].copy()
    filtered_reviews['cleaned_text'] = filtered_reviews['reviews_text'].apply(clean_text)
    
    try:
        X = tf_idf_vectorizer.transform(filtered_reviews['cleaned_text'])
    except ValueError:
        raise ValueError("Vectorizer and model dimension mismatch. Ensure the vectorizer used during training is used here.")

    filtered_reviews['positive_score'] = sentiment_model.predict_proba(X)[:, 1]
    sentiment_scores = filtered_reviews.groupby('name')['positive_score'].mean().reset_index()

    final = pd.merge(top_20, sentiment_scores, left_on='product', right_on='name').drop(columns=['name'])
    final['hybrid_score'] = 0.6 * final['score'] + 0.4 * final['positive_score']

    return final.sort_values(by='hybrid_score', ascending=False).head(final_n)

# ---------------------------
# STREAMLIT UI CONFIGURATION
# ---------------------------
st.set_page_config(page_title="🛍️ Smart Product Recommender", layout="centered")

st.markdown("""
    <style>
        .main-title {
            font-size: 40px;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
            margin-top: 10px;
        }
        .sub-title {
            font-size: 20px;
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 20px;
        }
        .recommend-box {
            background-color: #f0f8ff;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #d1e7dd;
            margin-top: 20px;
        }
        .footer {
            text-align: center;
            color: gray;
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🌟 Smart Product Recommender</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Get AI-powered suggestions using Collaborative Filtering + Sentiment Analysis</div>", unsafe_allow_html=True)

# ---------------------------
# INPUT SECTION
# ---------------------------
st.markdown("### 🔑 Enter User ID Below")
user_input = st.text_input("Example: 2784, 1499, 3134...", placeholder="Enter a valid User ID")

if st.button("🚀 Get Recommendations"):
    try:
        user_id = int(user_input)
        with st.spinner("🔍 Analyzing your preferences..."):
            results = recommend_products(user_id)

        if results.empty:
            st.warning("⚠️ No recommendations found for this User ID.")
        else:
            st.success("✅ Top 5 Product Recommendations just for you!")
            st.markdown("<div class='recommend-box'>", unsafe_allow_html=True)
            st.dataframe(results.reset_index(drop=True), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error occurred: {e}")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.markdown("<div class='footer'>Capstone Project | Streamlit Deployment | 2025</div>", unsafe_allow_html=True)
