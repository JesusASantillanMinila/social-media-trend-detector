import streamlit as st
import umap
import hdbscan
from sentence_transformers import SentenceTransformer
import requests
import pandas as pd
import os
from google import genai
import numpy as np
from pydantic import BaseModel, Field
from google.genai.types import GenerateContentConfig
import plotly.express as px

# --- Pydantic Models ---
class TrendInsight(BaseModel):
    trend_name: str = Field(description="A catchy 2-to-4 word label for the trend.")
    key_ingredients_or_products: list[str] = Field(description="Specific products, ingredients, or tools explicitly mentioned in the posts.")
    consumer_pain_point: str = Field(description="The underlying problem or insecurity the consumers are trying to solve.")
    capitalization_strategy: str = Field(description="A 1-sentence idea on how a brand could capitalize on this specific trend. Make it directand actionable.")
    actionability_score: int = Field(description="A score from 1-10 on how easily a business could monetize this trend.")

class KeywordResponse(BaseModel):
    keywords: list[str] = Field(description="A list of semantically related words.")

# --- Credentials ---


try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    BSKY_HANDLE = st.secrets["BSKY_HANDLE"]
    BSKY_APP_PASSWORD = st.secrets["BSKY_APP_PASSWORD"] 
except (KeyError, FileNotFoundError):
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    BSKY_HANDLE = os.environ.get("BSKY_HANDLE") 
    BSKY_APP_PASSWORD = os.environ.get("BSKY_APP_PASSWORD") 

if not GEMINI_API_KEY:
    st.warning("Please configure your environmental variables.")

# --- Functions ---
@st.cache_data(show_spinner=False)
def get_related_keywords(primary_word: str, num_related: int = 4) -> list[str]:
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"""
    You are an expert consumer behavior analyst focused on predicting emerging trends and discovering market whitespace. 
    I am building a search query to find organic, everyday conversations about '{primary_word}' on social media.
    
    Generate exactly {num_related} highly relevant search terms that capture the context where new trends emerge.
    
    To find whitespace BEFORE a trend happens, your terms must focus on:
    - Consumer pain points, struggles, or complaints (e.g., "damaged", "soreness", "too expensive")
    - Daily routines, habits, or generic goals (e.g., "morning routine", "hydration", "recovery")
    - Unmet needs or DIY workarounds
    
    STRICT RULES:
    - ABSOLUTELY NO brand names or specific product names.
    - ABSOLUTELY NO existing viral trend names or catchy social media slang (we want the raw behaviors that precede trends).
    - STAY DOMAIN-ANCHORED: Keep terms strictly within the direct ecosystem of '{primary_word}'. Do not drift into broad, unrelated macro-topics.
    - Keep terms short (1 to 2 words maximum).
    - Do NOT include the '#' symbol.
    """
    try:
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite', 
            contents=prompt,
            config=GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=KeywordResponse,
            )
        )
        return response.parsed.keywords[:num_related]
    except Exception as e:
        return []

@st.cache_data(show_spinner=False)
def fetch_bluesky_posts(query, target_count):
    session_url = "https://bsky.social/xrpc/com.atproto.server.createSession"
    session_data = {"identifier": BSKY_HANDLE, "password": BSKY_APP_PASSWORD}
    session_resp = requests.post(session_url, json=session_data).json()
    
    if "accessJwt" not in session_resp:
        st.error("Failed to authenticate with Bluesky.")
        return pd.DataFrame()
        
    auth_token = session_resp["accessJwt"]
    headers = {"Authorization": f"Bearer {auth_token}"}
    search_url = "https://bsky.social/xrpc/app.bsky.feed.searchPosts"
    
    posts_data = []
    cursor = None
    
    while len(posts_data) < target_count:
        params = {"q": query, "limit": 100} 
        if cursor:
            params["cursor"] = cursor
            
        resp = requests.get(search_url, headers=headers, params=params).json()
        new_posts = resp.get("posts", [])
        
        if not new_posts:
            break 
            
        for post in new_posts:
            posts_data.append({
                "text": post["record"]["text"],
                "created_at": post["record"]["createdAt"],
                "author": post["author"]["handle"],
                "replyCount": post.get("replyCount", 0),
                "repostCount": post.get("repostCount", 0),
                "likeCount": post.get("likeCount", 0),
                "quoteCount": post.get("quoteCount", 0)
            })
            
        cursor = resp.get("cursor")
        if not cursor:
            break
            
    return pd.DataFrame(posts_data[:target_count])

def filter_spam_posts(df, threshold):
    df_scored = df.copy()
    df_scored['spam_score'] = 0.0
    
    is_duplicate = df_scored.duplicated(subset=['text'], keep='first')
    df_scored.loc[is_duplicate, 'spam_score'] += 0.5
    
    df_scored['total_engagement'] = (
        df_scored['replyCount'] + 
        df_scored['repostCount'] + 
        df_scored['likeCount'] + 
        df_scored['quoteCount']
    )
    df_scored.loc[df_scored['total_engagement'] == 0, 'spam_score'] += 0.2
    
    df_scored['hashtag_count'] = df_scored['text'].str.count(r'#\w+')
    df_scored['link_count'] = df_scored['text'].str.count(r'http[s]?://')
    
    df_scored.loc[df_scored['hashtag_count'] > 4, 'spam_score'] += 0.15
    df_scored.loc[df_scored['link_count'] >= 2, 'spam_score'] += 0.15
    df_scored['spam_score'] = df_scored['spam_score'].clip(upper=1.0)
    
    df_filtered = df_scored[df_scored['spam_score'] < threshold].copy()
    spam_count = len(df_scored) - len(df_filtered)
    
    df_filtered = df_filtered.drop(columns=['total_engagement', 'hashtag_count', 'link_count'])
    return df_filtered, spam_count

@st.cache_resource(show_spinner=False)
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def cluster_social_posts(df, cluster_fraction, sample_fraction):
    model = load_sentence_model()
    embeddings = model.encode(df['text'].tolist())
    
    # 5D UMAP for HDBSCAN clustering
    umap_model = umap.UMAP(n_neighbors=30, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    reduced_embeddings = umap_model.fit_transform(embeddings)
    
    total_posts = len(df)
    dynamic_min_cluster_size = max(5, int(total_posts * cluster_fraction))
    dynamic_min_samples = max(5, int(dynamic_min_cluster_size * 0.5))
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=dynamic_min_cluster_size, min_samples=dynamic_min_samples, metric='euclidean')
    df['cluster_id'] = clusterer.fit_predict(reduced_embeddings)
    
    # 2D UMAP for visualization purposes
    umap_2d = umap.UMAP(n_components=2, min_dist=0.0, metric='cosine', random_state=42)
    embeddings_2d = umap_2d.fit_transform(embeddings)
    df['x'] = embeddings_2d[:, 0]
    df['y'] = embeddings_2d[:, 1]
    
    return df[df['cluster_id'] != -1]

def extract_actionable_insights(df_clustered, seed_keyword):
    client = genai.Client(api_key=GEMINI_API_KEY)
    cluster_insights = {}
    cluster_sizes = df_clustered['cluster_id'].value_counts()
    top_6_clusters = cluster_sizes.head(6).index.tolist()
    
    for cluster_id in top_6_clusters:
        cluster_df = df_clustered[df_clustered['cluster_id'] == cluster_id].copy()
        cluster_df['total_interactions'] = (
            cluster_df['likeCount'] + cluster_df['repostCount'] + 
            cluster_df['replyCount'] + cluster_df['quoteCount']
        )
        sample_posts = cluster_df.sort_values(by='total_interactions', ascending=False)['text'].head(15).tolist()
        posts_text = "\n- ".join(sample_posts)
        
        prompt = f"""
        You are an expert consumer trend analyst and product developer. 
        The following social media posts are all centered around the core topic of '{seed_keyword}'. 
        They have been clustered together because they share a specific emerging theme, behavior, or pain point:
        - {posts_text}
        Extract the underlying trend from this specific cluster and identify exactly how a business in the '{seed_keyword}' space can capitalize on it.
        """
        
        try:
            response = client.models.generate_content(
                model='gemini-3.5-flash-lite', 
                contents=prompt,
                config=GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=TrendInsight,
                )
            )
            cluster_insights[cluster_id] = response.parsed
        except Exception:
            continue
            
    df_clustered['trend_name'] = df_clustered['cluster_id'].map(lambda x: cluster_insights[x].trend_name if x in cluster_insights else None)
    df_clustered['key_products'] = df_clustered['cluster_id'].map(lambda x: ", ".join(cluster_insights[x].key_ingredients_or_products) if x in cluster_insights else None)
    df_clustered['pain_point'] = df_clustered['cluster_id'].map(lambda x: cluster_insights[x].consumer_pain_point if x in cluster_insights else None)
    df_clustered['strategy'] = df_clustered['cluster_id'].map(lambda x: cluster_insights[x].capitalization_strategy if x in cluster_insights else None)
    df_clustered['actionability_score'] = df_clustered['cluster_id'].map(lambda x: cluster_insights[x].actionability_score if x in cluster_insights else None)
    
    return df_clustered

# --- Streamlit UI ---
st.set_page_config(page_title="Trend Whitespace Analyzer", layout="wide")
st.title("Trend Whitespace Analyzer")

# Top Inputs
col1, col2 = st.columns(2)
with col1:
    seed_keyword = st.text_input("Initial Keyword", "world cup")
with col2:
    spam_threshold = st.slider("Spam Filter Threshold", 0.0, 1.0, 0.1)

if st.button("Run Analysis"):
    if not BSKY_HANDLE or not GEMINI_API_KEY:
        st.error("Please ensure your BSKY_HANDLE, BSKY_APP_PASSWORD, and GEMINI_API_KEY environment variables are set.")
        st.stop()

    with st.spinner("Fetching data and analyzing trends..."):
        # 1. Background Keyword Expansion (Hidden from user)
        related_words = get_related_keywords(seed_keyword, num_related=3)
        all_keywords = [seed_keyword] + [item.replace(" ", "") for item in related_words]
        
        # 2. Fetch Posts
        all_posts = []
        for keyword in all_keywords:
            df_temp = fetch_bluesky_posts(keyword, target_count=500) # Lowered for speed in UI, adjust as needed
            all_posts.append(df_temp)
            
        df_combined = pd.concat(all_posts, ignore_index=True)
        total_fetched = len(df_combined)
        
        # 3. Filter Spam
        df_clean, spam_removed_count = filter_spam_posts(df_combined, threshold=spam_threshold)
        
        # 4. Cluster & Extract
        if not df_clean.empty:
            df_clustered = cluster_social_posts(df_clean, cluster_fraction=0.01, sample_fraction=0.002)
            df_labeled = extract_actionable_insights(df_clustered, seed_keyword)
            
            # --- Layout Tabs ---
            tab1, tab2 = st.tabs(["Clustering & Results", "Spam Metrics"])
            
            with tab1:
                st.subheader("Cluster Visualization")
                # Scatter plot using the 2D UMAP coordinates
                df_viz = df_labeled.dropna(subset=['trend_name'])
                if not df_viz.empty:
                    fig = px.scatter(
                        df_viz, x='x', y='y', color='trend_name', 
                        hover_data=['text'], title="Post Clusters via UMAP",
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data to map trends.")

                st.subheader("Extracted Trends & Strategies")
                # Group by cluster ID to show unique insights
                unique_insights = df_labeled.drop_duplicates(subset=['cluster_id']).dropna(subset=['trend_name'])
                display_cols = ['trend_name', 'key_products', 'pain_point', 'strategy', 'actionability_score']
                st.dataframe(unique_insights[display_cols], use_container_width=True)

            with tab2:
                st.subheader("Spam Filter Metrics")
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Total Posts Fetched", total_fetched)
                col_m2.metric("Spam Posts Removed", spam_removed_count)
                col_m3.metric("Clean Posts Analyzed", len(df_clean))
                
        else:
            st.warning("No posts remaining after spam filtering.")