import streamlit as st
import pandas as pd
import pickle

# ---------------------------
# SETUP: Load Models and Data
# ---------------------------
tf_idf_vectorizer = pickle.load(open("models/tf_idf_vectorizer.pkl", "rb"))
sentiment_model = pickle.load(open("models/sentiment_model.pkl", "rb"))
item_similarity_df = pickle.load(open("models/similarity_matrix.pkl", "rb"))
review_df = pd.read_csv("data/review_df.csv")

# ---------------------------
# HELPER FUNCTION: Text Cleaner
# ---------------------------
def clean_text(text):
    return text.lower()

# ---------------------------
# CORE FUNCTION: Generate Recommendations
# ---------------------------
def get_final_recommendations(user_id, top_n=20, final_n=5):
    user_item_matrix = pd.pivot_table(review_df, index='user_id', columns='name', values='reviews_rating')
    if user_id not in user_item_matrix.index:
        return pd.DataFrame()

    user_ratings = user_item_matrix.loc[user_id].dropna()
    scores = pd.Series(dtype=float)

    for item, rating in user_ratings.items():
        if item in item_similarity_df:
            similar_items = item_similarity_df[item]
            scores = scores.add(similar_items * rating, fill_value=0)

    scores = scores.drop(user_ratings.index, errors='ignore')
    top_20_df = scores.sort_values(ascending=False).head(top_n).reset_index()
    top_20_df.columns = ['product', 'score']

    filtered_reviews = review_df[review_df['name'].isin(top_20_df['product'])].copy()
    filtered_reviews['cleaned_text'] = filtered_reviews['reviews_text'].apply(clean_text)
    X_reviews = tfidf_vectorizer.transform(filtered_reviews['cleaned_text'])
    filtered_reviews['positive_score'] = sentiment_model.predict_proba(X_reviews)[:, 1]

    avg_sentiment = filtered_reviews.groupby('name')['positive_score'].mean().reset_index()
    final_recs = pd.merge(top_20_df, avg_sentiment, left_on='product', right_on='name')
    final_recs = final_recs.drop(columns=['name'])
    final_recs['hybrid_score'] = 0.6 * final_recs['score'] + 0.4 * final_recs['positive_score']

    return final_recs.sort_values(by='hybrid_score', ascending=False).head(final_n)

# ---------------------------
# STREAMLIT UI CONFIGURATION
# ---------------------------
st.set_page_config(
    page_title="Smart Product Recommender",
    page_icon="🛍️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main-title {
        font-size:40px !important;
        font-weight:bold;
        color:#2c3e50;
        text-align:center;
        padding: 10px 0 0 0;
    }
    .sub-title {
        font-size:20px !important;
        color:#7f8c8d;
        text-align:center;
    }
    .recommend-box {
        background-color: #e8f6f3;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        border: 1px solid #d1e7dd;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌟 Smart Product Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Get AI-powered suggestions based on collaborative filtering + sentiment</div>', unsafe_allow_html=True)

# ---------------------------
# INPUT FORM
# ---------------------------
user_id = st.text_input("🔑 Enter User ID", placeholder="Try 2784, 1499, 3134...")
if st.button("🚀 Get Recommendations"):
    try:
        user_id = int(user_id)
        with st.spinner("Analyzing preferences..."):
            recs = get_final_recommendations(user_id)

        if recs.empty:
            st.warning("No recommendations found for this User ID.")
        else:
            st.markdown('<div class="recommend-box"><h4>✅ Top 5 Product Recommendations</h4>', unsafe_allow_html=True)
            st.dataframe(recs.reset_index(drop=True), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error occurred: {e}")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("""---""")
st.markdown(
    "<p style='text-align: center; color: gray;'>Capstone Project | Deployed via Streamlit</p>",
    unsafe_allow_html=True
)
