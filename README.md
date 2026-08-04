### Trend Whitespace Analyzer

This Streamlit application analyzes social media conversations to identify emerging market trends, consumer pain points, and business whitespace opportunities.

### Features

* Expands seed keywords into related search terms using Google Gemini.
* Fetches organic posts from the Bluesky social network.
* Filters out spam, duplicates, and low-value noise.
* Clusters semantic topics using UMAP and HDBSCAN via Sentence Transformers.
* Generates actionable business insights, including trend names, capitalization strategies, and actionability scores.
* Visualizes data using an interactive Plotly scatter map and pie chart.

### Prerequisites

You must configure the following environment variables or add them to your Streamlit secrets file:

* GEMINI_API_KEY: Your Google Gemini API key.
* BSKY_HANDLE: Your Bluesky account handle.
* BSKY_APP_PASSWORD: Your Bluesky app password.

### Usage

1. Install dependencies: streamlit, umap-learn, hdbscan, sentence-transformers, requests, pandas, google-genai, numpy, pydantic, and plotly.
2. Run the application via Streamlit.
3. Enter a core topic into the web interface.
4. Adjust the spam filter strictness slider.
5. Click the Run Analysis button to view the trend map and whitespace opportunities.
