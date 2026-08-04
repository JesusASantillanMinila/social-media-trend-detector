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
import plotly.graph_objects as go
import time

# --- Pydantic Models ---
class TrendInsight(BaseModel):
    trend_name: str = Field(description="A catchy 2-to-4 word label for the trend.")
    key_ingredients_or_products: list[str] = Field(description="Specific products, ingredients, or tools explicitly mentioned in the posts.")
    consumer_pain_point: str = Field(description="The underlying problem or insecurity the consumers are trying to solve.")
    capitalization_strategy: str = Field(description="A 1-sentence idea on how a brand could capitalize on this specific trend. Make it directand actionable.")
    actionability_score: int = Field(description="A score from 1-10 on how easily a business could monetize this trend.")

class KeywordResponse(BaseModel):
    keywords: list[str] = Field(description="A list of semantically related words.")

# --- Streamlit Page Config ---
st.set_page_config(page_title="Trend Whitespace Analyzer", page_icon="✨", layout="wide")

# --- Credentials ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    BSKY_HANDLE = st.secrets["BSKY_HANDLE"]
    BSKY_APP_PASSWORD = st.secrets["BSKY_APP_PASSWORD"] 
except (KeyError, FileNotFoundError):
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    BSKY_HANDLE = os.environ.get("BSKY_HANDLE") 
    BSKY_APP_PASSWORD = os.environ.get("BSKY_APP_PASSWORD") 

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
    
    umap_model = umap.UMAP(n_neighbors=30, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    reduced_embeddings = umap_model.fit_transform(embeddings)
    
    total_posts = len(df)
    dynamic_min_cluster_size = max(5, int(total_posts * cluster_fraction))
    dynamic_min_samples = max(5, int(dynamic_min_cluster_size * 0.5))
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=dynamic_min_cluster_size, min_samples=dynamic_min_samples, metric='euclidean')
    df['cluster_id'] = clusterer.fit_predict(reduced_embeddings)
    
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

# --- UI Layout ---
st.title("✨ Trend Whitespace Analyzer")
st.markdown("Discover emerging market whitespace and hidden consumer pain points from everyday social conversations.")

if not GEMINI_API_KEY:
    st.warning("⚠️ Please configure your environmental variables to proceed.")

# --- Upperbar Control Panel ---
with st.container(border=True):
    col1, col2, col3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    
    with col1:
        seed_keyword = st.text_input("Core Topic / Keyword", "world cup")
    with col2:
        spam_threshold = st.slider("Spam Filter Strictness", 0.0, 1.0, 0.1, help="Lower values filter out more spam.")
    with col3:
        run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Main Execution
if run_btn:
    if not BSKY_HANDLE or not GEMINI_API_KEY:
        st.error("Missing credentials. Please check your environment variables.")
        st.stop()

    # Step-by-Step Status Container
    with st.status("Initializing Trend Engine...", expanded=True) as status:
        
        # 1. Background Keyword Expansion
        st.write("🧠 Contextualizing keyword...")
        related_words = get_related_keywords(seed_keyword, num_related=3)
        all_keywords = [seed_keyword] + [item.replace(" ", "") for item in related_words]
        # st.markdown(f"**Expanded Search Vectors:** `{', '.join(all_keywords)}`")
        
        # 2. Fetch Posts incrementally
        st.write("📡 Fetching raw social signals...")
        progress_bar = st.progress(0)
        all_posts = []
        
        for idx, keyword in enumerate(all_keywords):
            df_temp = fetch_bluesky_posts(keyword, target_count=500)
            all_posts.append(df_temp)
            progress_bar.progress((idx + 1) / len(all_keywords))
            
        df_combined = pd.concat(all_posts, ignore_index=True)
        total_fetched = len(df_combined)
        progress_bar.empty()
        
        # 3. Filter Spam
        st.write("🧹 Scrubbing spam and low-value noise...")
        df_clean, spam_removed_count = filter_spam_posts(df_combined, threshold=spam_threshold)
        
        # 4. Cluster & Extract
        if not df_clean.empty:
            st.write("🌌 Mapping semantic clusters using UMAP & HDBSCAN...")
            df_clustered = cluster_social_posts(df_clean, cluster_fraction=0.01, sample_fraction=0.002)
            
            st.write("🤖 Generating actionable business insights with Gemini...")
            df_labeled = extract_actionable_insights(df_clustered, seed_keyword)
            
            status.update(label="Analysis Complete!", state="complete", expanded=False)
        else:
            status.update(label="Process Halted: No data remained after filtering.", state="error")
            st.stop()

    # --- Final Results Layout ---
    st.success(f"Successfully processed {total_fetched} posts and extracted key insights!")
    
    col_map, col_health = st.columns([2, 1])
    
with col_map:
            st.subheader("📊 Semantic Trend Map")
            df_viz = df_labeled.dropna(subset=['trend_name'])
            if not df_viz.empty:
                
                fig = px.scatter(
                    df_viz, x='x', y='y', color='trend_name', 
                    hover_data={'x': False, 'y': False, 'text': True}, 
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
                
                
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", 
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(
                        showline=True,        
                        showticklabels=False, 
                        title="",             
                        showgrid=False,       
                        zeroline=False        
                    ),
                    yaxis=dict(
                        showline=True,        
                        showticklabels=False, 
                        title="",             
                        showgrid=False,       
                        zeroline=False        
                    ),
                    margin=dict(t=20, b=20, l=10, r=10)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough cohesive data to map visual trends.")

    with col_health:
        st.subheader("📈 Data Health & Noise Breakdown")
        
        # Clean Data Percentage Calculation
        clean_pct = round((len(df_clean) / total_fetched * 100), 1) if total_fetched > 0 else 0
        
        # Stylized Modern Donut Chart
        fig_spam = go.Figure(data=[go.Pie(
            labels=['Clean Posts', 'Spam / Noise'], 
            values=[len(df_clean), spam_removed_count],
            hole=0.68,
            marker=dict(
                colors=['#6366F1', '#EC4899'], # Modern Indigo & Rose
                line=dict(color='rgba(255,255,255,0.2)', width=2)
            ),
            hoverinfo='label+value+percent',
            textinfo='none',
            insidetextorientation='radial'
        )])

        fig_spam.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            margin=dict(t=10, b=30, l=10, r=10),
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[
                dict(
                    text=f"<b>{clean_pct}%</b><br><span style='font-size:12px;color:gray;'>Clean Yield</span>",
                    x=0.5, y=0.5,
                    font=dict(size=22),
                    showarrow=False
                )
            ]
        )
        
        st.plotly_chart(fig_spam, use_container_width=True)
        
        # Summary Metrics beneath the chart
        m1, m2 = st.columns(2)
        m1.metric("Total Fetched", total_fetched)
        m2.metric("Spam Removed", spam_removed_count)

    st.divider()
    st.subheader("💡 Actionable Whitespace Opportunities")
    
    unique_insights = df_labeled.drop_duplicates(subset=['cluster_id']).dropna(subset=['trend_name']).sort_values(by='actionability_score', ascending=False)
    
    if not unique_insights.empty:
        for _, row in unique_insights.iterrows():
            with st.container(border=True):
                st.markdown(f"### 🔥 {row['trend_name']}")
                
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**Consumer Pain Point:** {row['pain_point']}")
                    st.markdown(f"**Capitalization Strategy:** {row['strategy']}")
                    st.markdown(f"**Key Products/Ingredients:** `{row['key_products']}`")
                with c2:
                    st.metric("Actionability Score", f"{row['actionability_score']} / 10")
    else:
        st.write("No distinct insights could be generated from this dataset.")