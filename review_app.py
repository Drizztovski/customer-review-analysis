"""
review_app.py — Streamlit UI for Customer Review Analysis (Student Version)
===============================================================================
A web interface for analyzing Amazon reviews using traditional ML and modern AI
approaches. Perform sentiment classification, extract topics, and generate insights.

Run with:  streamlit run review_app.py

Students: Complete the TODO sections to build the full app.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Make sure we can import from the current directory
sys.path.insert(0, os.path.dirname(__file__))

# NOTE: Import from solution/ to use completed engine, or from current dir to test yours
# from solution.review_analyzer import ReviewAnalyzer
from review_analyzer import ReviewAnalyzer


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Customer Review Analysis",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📝 Customer Review Analysis Dashboard")
st.caption("Analyze Amazon reviews with traditional ML and modern AI")


# ============================================================
# SIDEBAR: Configuration
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "API Key (optional)",
        type="password",
        help="Needed for LLM features (Gemini)",
    )

    data_path = st.text_input(
        "Data file path",
        value="data/amazon_reviews.csv",
    )

    n_llm_samples = st.slider(
        "Number of LLM samples",
        min_value=5, max_value=50, value=20,
        help="How many reviews to analyze with the LLM"
    )

    run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    st.divider()

    if st.session_state.get('analysis_run'):
        st.markdown("**📈 ML Results**")
        from sklearn.metrics import accuracy_score, classification_report
        _engine = st.session_state.engine
        _accuracy = accuracy_score(_engine.y_test, _engine.y_pred)
        st.metric("Accuracy", f"{_accuracy:.1%}")
        st.markdown("**Classification Report**")
        _report = classification_report(_engine.y_test, _engine.y_pred)
        st.code(_report, language=None)


# ============================================================
# SESSION STATE
# ============================================================

if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False


# ============================================================
# RUN ANALYSIS
# ============================================================

if run_button:
    with st.spinner("Running analysis..."):
        try:
            engine = ReviewAnalyzer(
                filepath=data_path,
                api_key=api_key if api_key else None
            )
            engine.run_analysis()
            st.session_state.engine = engine
            st.session_state.analysis_run = True
            st.toast("Analysis complete!", icon="✅")
        except Exception as e:
            st.error(f"Error running analysis: {e}")


# ============================================================
# MAIN AREA: Results Display
# ============================================================

if st.session_state.analysis_run:
    engine = st.session_state.engine
    summary = engine.get_summary()

    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🤖 ML Results", "✨ AI Insights"])

    # --------------------------------------------------------
    # TAB 1: Overview
    # --------------------------------------------------------
    with tab1:
        st.subheader("Dataset Overview")

        # Metrics row
        sentiment_dist = summary['sentiment_distribution']
        total = summary['total_reviews']
        pct_negative = sentiment_dist.get('Negative', 0) / total * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reviews", f"{total:,}")
        col2.metric("Avg Rating", f"{summary['avg_rating']:.2f} ⭐")
        col3.metric("% Negative", f"{pct_negative:.1f}%")
        col4.metric("Avg Word Count", f"{summary['avg_word_count']:.0f}")

        st.divider()

        # Charts side by side
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Rating Distribution**")
            rating_counts = engine.df['rating'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(rating_counts.index, rating_counts.values, color='skyblue', edgecolor='black')
            ax.set_xlabel('Rating')
            ax.set_ylabel('Count')
            ax.set_title('Reviews by Star Rating')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            plt.close()

        with col_right:
            st.markdown("**Sentiment Distribution**")
            sent_counts = engine.df['sentiment_label'].value_counts()
            colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}
            bar_colors = [colors.get(s, '#888888') for s in sent_counts.index]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(sent_counts.index, sent_counts.values, color=bar_colors, edgecolor='black')
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Count')
            ax.set_title('Reviews by Sentiment')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            plt.close()

    # --------------------------------------------------------
    # TAB 2: ML Results
    # --------------------------------------------------------
    with tab2:
        st.subheader("Machine Learning Results")

        from sklearn.metrics import classification_report, accuracy_score

        accuracy = accuracy_score(engine.y_test, engine.y_pred)
        st.metric("Classifier Accuracy", f"{accuracy:.1%}")

        st.divider()

        st.markdown("**Classification Report**")
        report = classification_report(engine.y_test, engine.y_pred)
        st.code(report)

        st.divider()

        st.markdown("**Try Your Own Review**")
        custom_review = st.text_area("Enter a review to classify:", placeholder="Type a review here...")
        if st.button("🔍 Classify Review"):
            if custom_review.strip():
                from review_analyzer import clean_text
                cleaned = clean_text(custom_review)
                vectorized = engine.vectorizer.transform([cleaned])
                prediction = engine.classifier.predict(vectorized)[0]
                color = "green" if prediction == "Positive" else "red"
                st.markdown(f"**Predicted Sentiment:** :{color}[{prediction}]")
            else:
                st.warning("Please enter a review first.")

        st.divider()

        st.markdown("**Top Words by TF-IDF Score**")
        fig, ax = plt.subplots(figsize=(10, 5))
        feature_names = np.array(engine.vectorizer.get_feature_names_out())
        tfidf_scores = np.asarray(engine.tfidf_matrix.mean(axis=0)).ravel()
        top_indices = np.argsort(tfidf_scores)[-15:][::-1]
        top_words = feature_names[top_indices]
        top_scores = tfidf_scores[top_indices]
        ax.barh(range(len(top_words)), top_scores, color='coral', edgecolor='black')
        ax.set_yticks(range(len(top_words)))
        ax.set_yticklabels(top_words)
        ax.set_xlabel('Average TF-IDF Score')
        ax.set_title('Top 15 Words by TF-IDF')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.divider()

    # --------------------------------------------------------
    # TAB 3: AI Insights
    # --------------------------------------------------------
    with tab3:
        st.subheader("AI-Powered Insights")

        if not api_key:
            st.info("Enter a Gemini API key in the sidebar to enable AI features.")
        else:
            col_a, col_b, col_c = st.columns(3)

            if col_a.button("🔄 Compare ML vs LLM", use_container_width=True):
                with st.spinner("Classifying reviews with Gemini..."):
                    comparison_df = engine.classify_reviews_llm(n_llm_samples)
                    st.markdown("**ML vs LLM Classification Comparison**")
                    st.dataframe(comparison_df, use_container_width=True)
                    ml_acc = comparison_df['ml_match'].mean()
                    llm_acc = comparison_df['llm_match'].mean()
                    m1, m2 = st.columns(2)
                    m1.metric("ML Accuracy on Sample", f"{ml_acc:.1%}")
                    m2.metric("LLM Accuracy on Sample", f"{llm_acc:.1%}")

            if col_b.button("🏷️ Extract Topics", use_container_width=True):
                with st.spinner("Extracting topics with Gemini..."):
                    topics = engine.get_topics(n_llm_samples)
                    st.markdown("**Top Themes Across Reviews**")
                    for topic in topics:
                        with st.expander(f"📌 {topic.get('topic', 'Unknown')} — {topic.get('frequency_hint', '')}"):
                            st.markdown(f"*\"{topic.get('example_quote', '')}\"*")

            if col_c.button("🔎 Aspect Analysis", use_container_width=True):
                with st.spinner("Running aspect analysis with Gemini..."):
                    aspects_results = engine.extract_all_aspects(n_llm_samples)
                    st.markdown("**Aspect-Level Sentiment per Review**")
                    for result in aspects_results:
                        with st.expander(f"📝 {result['review_text']}"):
                            for aspect in result['aspects']:
                                sentiment = aspect.get('sentiment', '')
                                color = "green" if sentiment == "positive" else ("red" if sentiment == "negative" else "gray")
                                st.markdown(f"- **{aspect.get('aspect', '?')}** — :{color}[{sentiment}]")
                                st.caption(f"*\"{aspect.get('quote', '')}\"*")

else:
    st.info("Configure your settings in the sidebar and click **🚀 Run Analysis** to get started.")


# ============================================================
# OPTIONAL EXTENSIONS
# ============================================================

# OPTIONAL TODO A: Add word cloud visualization
# Show a word cloud of the most common words in reviews

# OPTIONAL TODO B: Add rating vs sentiment breakdown
# Create a cross-tab showing how ratings (1-5) map to ML sentiment predictions

# OPTIONAL TODO C: Add time series analysis
# If review_date is available, show review count over time
