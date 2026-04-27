"""
review_analyzer.py — Amazon Product Review Analysis Engine

Analyzes Amazon product reviews using both traditional machine learning
(TF-IDF + Logistic Regression) and modern LLM-based approaches (Google Gemini
zero-shot classification, aspect extraction, and topic modeling).

Dataset: data/amazon_reviews.csv
Columns: review_id, reviewer_name, country, review_date, rating, review_title,
         review_text, date_of_experience

Author: AJ Amatrudo
"""

import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Optional: Google Gemini for LLM features
try:
    from google import genai
except ImportError:
    genai = None


# ============================================================================
# PART A: Data Loading & Cleaning
# ============================================================================

def load_and_clean(filepath):
    """
    Load CSV file and perform initial data cleaning.

    Drops rows with missing review_text or rating, strips whitespace from
    string columns, and parses review_date as datetime.

    Parameters
    ----------
    filepath : str
        Path to the CSV file (e.g., 'data/amazon_reviews.csv')

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with review_date as datetime, no missing review_text/rating

    Raises
    ------
    FileNotFoundError
        If filepath does not exist
    """
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['review_text', 'rating'])
    str_cols = df.select_dtypes(include='object').columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())
    df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
    return df


# ============================================================================
# PART B: Text Preprocessing
# ============================================================================

def clean_text(text):
    """
    Clean and normalize text for analysis.

    Converts to lowercase, removes punctuation and special characters,
    and strips extra whitespace.

    Parameters
    ----------
    text : str
        Raw text to clean

    Returns
    -------
    str
        Cleaned text in lowercase with normalized whitespace

    Example
    -------
    >>> clean_text("Hello, World!!!  How are you?")
    'hello world how are you'
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_reviews(df):
    """
    Preprocess review DataFrame with text cleaning and sentiment labeling.

    Applies clean_text() to the review_text column, adds a word_count column,
    and maps star ratings to sentiment labels: 1-2 = Negative, 3 = Neutral,
    4-5 = Positive.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with review_text and rating columns

    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned review_text, word_count, and sentiment_label columns
    """
    df['clean_text'] = df['review_text'].apply(clean_text)
    df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
    df['sentiment_label'] = df['rating'].apply(
        lambda r: 'Negative' if r <= 2 else ('Neutral' if r == 3 else 'Positive')
    )
    return df


# ============================================================================
# PART C: Traditional ML — TF-IDF + Classification
# ============================================================================

def build_tfidf_features(texts, max_features=2000):
    """
    Build TF-IDF feature matrix from text documents.

    Creates a TfidfVectorizer with English stopwords removed, fits it on the
    provided texts, and returns both the feature matrix and the fitted vectorizer.

    Parameters
    ----------
    texts : list or pd.Series
        List/Series of text documents
    max_features : int, optional
        Maximum number of features to extract (default: 2000)

    Returns
    -------
    tuple
        (sparse TF-IDF matrix, fitted TfidfVectorizer object)

    Example
    -------
    >>> matrix, vec = build_tfidf_features(texts, max_features=1000)
    >>> print(matrix.shape)  # (n_documents, 1000)
    """
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer


def train_sentiment_classifier(X_train, y_train):
    """
    Train a Logistic Regression classifier for sentiment prediction.

    Trains a Logistic Regression classifier with max_iter=1000 and
    random_state=42 on the provided TF-IDF features and sentiment labels.

    Parameters
    ----------
    X_train : sparse matrix or array-like
        TF-IDF feature matrix for training (from build_tfidf_features)
    y_train : array-like
        Sentiment labels for training ('Positive', 'Negative', 'Neutral')

    Returns
    -------
    LogisticRegression
        Fitted sentiment classifier model
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


# ============================================================================
# PART D: LLM-Based Analysis (Google Gemini)
# ============================================================================

def classify_sentiment_llm(review_text, client, model='gemini-2.5-flash'):
    """
    Classify sentiment of a review using Google Gemini zero-shot classification.

    Sends the review text to Gemini with a zero-shot prompt and returns one
    of three labels: 'Positive', 'Negative', or 'Neutral'. Defaults to
    'Neutral' if the response is unexpected.

    Parameters
    ----------
    review_text : str
        The review text to classify
    client : google.genai.Client
        Initialized Gemini API client
    model : str, optional
        Model name (default: 'gemini-2.5-flash')

    Returns
    -------
    str
        One of: 'Positive', 'Negative', or 'Neutral'

    Note
    ----
    Uses zero-shot classification (no training examples provided).
    """
    prompt = (
        f"Classify the sentiment of this review as exactly one of: "
        f"'Positive', 'Negative', or 'Neutral'. "
        f"Respond with only the single word label.\n\nReview: {review_text}"
    )
    response = client.models.generate_content(model=model, contents=prompt)
    result = response.text.strip()
    if result not in ('Positive', 'Negative', 'Neutral'):
        return 'Neutral'
    return result


def extract_aspects_llm(review_text, client, model='gemini-2.5-flash'):
    """
    Extract key aspects and their sentiments from a review using Gemini.

    Asks Gemini to identify product features or topics mentioned in the review
    and classify each as positive, negative, or neutral. Returns structured
    JSON parsed into a list of dicts. Empty list returned if JSON parsing fails.

    Parameters
    ----------
    review_text : str
        The review text to analyze
    client : google.genai.Client
        Initialized Gemini API client
    model : str, optional
        Model name (default: 'gemini-2.5-flash')

    Returns
    -------
    list
        List of dicts with keys: aspect, sentiment, quote
        Empty list if extraction fails

    Example
    -------
    >>> aspects = extract_aspects_llm("Great battery but poor screen", client)
    >>> print(aspects)
    [
        {'aspect': 'battery', 'sentiment': 'positive', 'quote': 'Great battery'},
        {'aspect': 'screen', 'sentiment': 'negative', 'quote': 'poor screen'}
    ]
    """
    prompt = (
        f"Analyze this review and extract the key aspects mentioned. "
        f"Return ONLY a JSON array with no markdown, no explanation. "
        f"Each item must have keys: aspect, sentiment (positive/negative/neutral), quote.\n\n"
        f"Review: {review_text}"
    )
    response = client.models.generate_content(model=model, contents=prompt)
    try:
        text = response.text.strip()
        text = re.sub(r'```json|```', '', text).strip()
        return json.loads(text)
    except json.JSONDecodeError:
        return []


# ============================================================================
# PART E: Topic & Insight Extraction
# ============================================================================

def extract_topics_llm(reviews_sample, client, model='gemini-2.5-flash'):
    """
    Extract top themes and topics from a sample of reviews using Gemini.

    Sends a batch of review texts to Gemini and asks it to identify the most
    common themes across all of them. Returns structured JSON parsed into a
    list of dicts. Empty list returned if JSON parsing fails.

    Parameters
    ----------
    reviews_sample : list
        List of 10-20 review texts
    client : google.genai.Client
        Initialized Gemini API client
    model : str, optional
        Model name (default: 'gemini-2.5-flash')

    Returns
    -------
    list
        List of dicts with keys: topic, frequency_hint, example_quote
        Empty list if extraction fails

    Example
    -------
    >>> topics = extract_topics_llm(reviews[:15], client)
    >>> print(topics)
    [
        {
            'topic': 'battery life',
            'frequency_hint': 'very common',
            'example_quote': 'Battery lasts all day'
        },
        ...
    ]
    """
    reviews_text = "\n\n".join([f"Review {i+1}: {r}" for i, r in enumerate(reviews_sample)])
    prompt = (
        f"Analyze these customer reviews and identify the top recurring themes. "
        f"Return ONLY a JSON array with no markdown, no explanation. "
        f"Each item must have keys: topic, frequency_hint (very common/occasional/rare), example_quote.\n\n"
        f"{reviews_text}"
    )
    response = client.models.generate_content(model=model, contents=prompt)
    try:
        text = response.text.strip()
        text = re.sub(r'```json|```', '', text).strip()
        return json.loads(text)
    except json.JSONDecodeError:
        return []


# ============================================================================
# PART F: Visualization Helpers
# ============================================================================

def plot_sentiment_distribution(df):
    """
    Create a bar chart showing sentiment distribution.

    Counts sentiment_label values and displays a bar chart with consistent
    colors: green for Positive, red for Negative, gray for Neutral.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with sentiment_label column

    Returns
    -------
    None
        Displays the plot
    """
    sentiment_counts = df['sentiment_label'].value_counts()
    plt.figure(figsize=(10, 6))
    sentiment_counts.plot(kind='bar', color=['#2ecc71', '#e74c3c', '#95a5a6'], edgecolor='black')
    plt.title('Sentiment Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_rating_distribution(df):
    """
    Create a bar chart showing distribution of star ratings.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with rating column

    Returns
    -------
    None
        Displays the plot
    """
    rating_counts = df['rating'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    rating_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Distribution of Product Ratings', fontsize=14, fontweight='bold')
    plt.xlabel('Rating', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_top_words(vectorizer, tfidf_matrix, n=15):
    """
    Create a bar chart of top N words by TF-IDF score.
    
    Parameters
    ----------
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer
    tfidf_matrix : sparse matrix
        TF-IDF feature matrix
    n : int, optional
        Number of top words to display (default: 15)

    Returns
    -------
    None
        Displays the plot
    """
    # Get feature names and sum TF-IDF scores across all documents
    feature_names = np.array(vectorizer.get_feature_names_out())
    tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()

    # Get top n indices
    top_indices = np.argsort(tfidf_scores)[-n:][::-1]
    top_words = feature_names[top_indices]
    top_scores = tfidf_scores[top_indices]

    plt.figure(figsize=(12, 6))
    plt.barh(range(len(top_words)), top_scores, color='coral', edgecolor='black')
    plt.yticks(range(len(top_words)), top_words)
    plt.xlabel('Average TF-IDF Score', fontsize=12)
    plt.title(f'Top {n} Words by TF-IDF', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


# ============================================================================
# PART G: ReviewAnalyzer Class
# ============================================================================

class ReviewAnalyzer:
    """
    Main class for analyzing Amazon reviews using traditional ML and LLM approaches.

    This class orchestrates the entire review analysis pipeline: loading data,
    preprocessing, building TF-IDF features, training classifiers, and querying
    the Gemini API for advanced insights.

    Attributes
    ----------
    filepath : str
        Path to the CSV file
    api_key : str
        Google Gemini API key
    model : str
        Model name for Gemini API calls
    client : google.genai.Client
        Initialized Gemini API client
    df : pd.DataFrame, optional
        Loaded and preprocessed DataFrame
    tfidf_matrix : sparse matrix, optional
        TF-IDF feature matrix
    vectorizer : TfidfVectorizer, optional
        Fitted TF-IDF vectorizer
    classifier : LogisticRegression, optional
        Fitted sentiment classifier
    """

    def __init__(self, filepath, api_key=None, model='gemini-2.5-flash'):
        """
        Initialize the ReviewAnalyzer.

        Parameters
        ----------
        filepath : str
            Path to the CSV file
        api_key : str, optional
            Google Gemini API key. If None, reads from environment variable GOOGLE_API_KEY
        model : str, optional
            Model name for Gemini calls (default: 'gemini-2.5-flash')

        """
        self.filepath = filepath
        self.model = model
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')

        # Initialize Gemini client if API key available (LLM features are optional)
        if self.api_key and genai is not None:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

        # Initialize storage attributes
        self.df = None
        self.raw_df = None
        self.tfidf_matrix = None
        self.vectorizer = None
        self.classifier = None
        self.X_test_text = None
        self.y_test = None
        self.y_pred = None
        self.full_tfidf = None

    def run_analysis(self):
        """
        Run the complete review analysis pipeline.

        Loads and cleans data, preprocesses reviews, filters to binary sentiment
        (Positive/Negative only), builds TF-IDF features, trains the classifier,
        and evaluates on the test set. Results stored in instance attributes.

        Returns
        -------
        None
            Results stored in self.df, self.tfidf_matrix, self.vectorizer,
            self.classifier, self.y_test, self.y_pred
        """
        self.df = load_and_clean(self.filepath)
        self.df = preprocess_reviews(self.df)

        # Filter to binary sentiment only
        binary_df = self.df[self.df['sentiment_label'] != 'Neutral'].copy()

        # Train/test split
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            binary_df['clean_text'], binary_df['sentiment_label'],
            test_size=0.2, random_state=42
        )

        # Build TF-IDF and train classifier
        self.tfidf_matrix, self.vectorizer = build_tfidf_features(X_train_text)
        self.classifier = train_sentiment_classifier(self.tfidf_matrix, y_train)

        # Evaluate
        X_test_tfidf = self.vectorizer.transform(X_test_text)
        self.y_test = y_test
        self.y_pred = self.classifier.predict(X_test_tfidf)
        self.X_test_text = X_test_text

        accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"Analysis complete! Loaded {len(self.df)} reviews. Accuracy: {accuracy:.1%}")

    def classify_reviews_llm(self, n_samples=20):
        """
        Classify sentiment of a sample of reviews using both ML and LLM approaches.

        Parameters
        ----------
        n_samples : int, optional
            Number of reviews to classify (default: 20)

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - review_text: original review text
            - ml_sentiment: prediction from LogisticRegression
            - llm_sentiment: prediction from Gemini
            - actual_sentiment: ground truth from sentiment_label
            - match: whether ML and LLM agree (both columns)
        """
        if self.df is None or self.classifier is None:
            raise ValueError("Must run run_analysis() first")

        sample_indices = np.random.choice(len(self.df), min(n_samples, len(self.df)), replace=False)
        sample_df = self.df.iloc[sample_indices].copy()

        # ML predictions
        X_sample = self.vectorizer.transform(sample_df['clean_text'])
        ml_predictions = self.classifier.predict(X_sample)

        # LLM predictions
        llm_predictions = []
        for text in sample_df['review_text']:
            sentiment = classify_sentiment_llm(text, self.client, self.model)
            llm_predictions.append(sentiment)

        # Comparison
        result_df = pd.DataFrame({
            'review_text': sample_df['review_text'].values,
            'ml_sentiment': ml_predictions,
            'llm_sentiment': llm_predictions,
            'actual_sentiment': sample_df['sentiment_label'].values,
        })

        result_df['ml_match'] = result_df['ml_sentiment'] == result_df['actual_sentiment']
        result_df['llm_match'] = result_df['llm_sentiment'] == result_df['actual_sentiment']

        return result_df

    def extract_all_aspects(self, n_samples=10):
        """
        Extract aspects and sentiments from a sample of reviews.

        Parameters
        ----------
        n_samples : int, optional
            Number of reviews to analyze (default: 10)

        Returns
        -------
        list
            List of dicts with keys: review_text, aspects
            Each aspects is a list of dicts with keys: aspect, sentiment, quote
        """
        if self.df is None:
            raise ValueError("Must run run_analysis() first")

        sample_indices = np.random.choice(len(self.df), min(n_samples, len(self.df)), replace=False)
        sample_reviews = self.df.iloc[sample_indices]['review_text'].values

        results = []
        for review_text in sample_reviews:
            aspects = extract_aspects_llm(review_text, self.client, self.model)
            results.append({
                'review_text': review_text[:100] + '...' if len(review_text) > 100 else review_text,
                'aspects': aspects
            })

        return results

    def get_topics(self, n_samples=15):
        """
        Get top themes and topics from a sample of reviews.

        Parameters
        ----------
        n_samples : int, optional
            Number of reviews to analyze (default: 15)

        Returns
        -------
        list
            List of dicts with keys: topic, frequency_hint, example_quote
        """
        if self.df is None:
            raise ValueError("Must run run_analysis() first")

        sample_indices = np.random.choice(len(self.df), min(n_samples, len(self.df)), replace=False)
        sample_reviews = self.df.iloc[sample_indices]['review_text'].tolist()

        topics = extract_topics_llm(sample_reviews, self.client, self.model)
        return topics

    def get_summary(self):
        """
        Get summary statistics of the analysis.
        
        Returns
        -------
        dict
            Dictionary with keys:
            - total_reviews: number of reviews
            - avg_rating: average rating
            - avg_word_count: average words per review
            - sentiment_distribution: dict of sentiment counts
            - classifier_accuracy: accuracy on test set
            - model_name: Gemini model used
        """
        if self.df is None or self.classifier is None:
            raise ValueError("Must run run_analysis() first")

        accuracy = accuracy_score(self.y_test, self.y_pred)

        return {
            'total_reviews': len(self.df),
            'avg_rating': self.df['rating'].mean(),
            'avg_word_count': self.df['word_count'].mean(),
            'sentiment_distribution': self.df['sentiment_label'].value_counts().to_dict(),
            'classifier_accuracy': accuracy,
            'model_name': self.model,
        }


# ============================================================================
# Example Usage (for students to uncomment and try)
# ============================================================================

if __name__ == '__main__':
    analyzer = ReviewAnalyzer('data/amazon_reviews.csv')
    analyzer.run_analysis()
    print(analyzer.get_summary())
