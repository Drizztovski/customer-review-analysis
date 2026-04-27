# Project 4: Customer Review Analysis

Analyze 21,000+ real Amazon reviews using **traditional ML** (TF-IDF + Logistic Regression) and **modern AI** (Google Gemini zero-shot classification, aspect-based sentiment, topic extraction). Compare both approaches to understand the evolution of text analysis.

## Dataset

**Source:** Amazon Reviews Dataset (Kaggle) — real customer reviews  
**Size:** 21,000+ reviews  
**File:** `data/amazon_reviews.csv`

| Column | Description |
|--------|-------------|
| `review_id` | Unique review identifier |
| `reviewer_name` | Reviewer display name |
| `country` | Reviewer country (US, GB, CA, IN, etc.) |
| `review_date` | Date review was posted |
| `rating` | Star rating (1-5) |
| `review_title` | Short review headline |
| `review_text` | Full review text |
| `date_of_experience` | Date of the experience reviewed |


## Traditional vs Modern AI Approach

| Approach | How It Works | Pros | Cons |
|----------|-------------|------|------|
| **TF-IDF + Logistic Regression** | Convert text to numerical features, train a classifier on labeled data | Fast, cheap at scale, deterministic | Needs labeled data, misses nuance |
| **Gemini Zero-Shot** | Describe the task in English, the LLM classifies without training | No training data needed, understands context | API cost, slower, non-deterministic |
| **Aspect-Based (Gemini)** | Ask the LLM to extract what aspects are positive/negative | Rich insights, no training | Expensive at scale |
| **Topic Extraction (Gemini)** | Ask the LLM to find common themes across reviews | Discovers structure automatically | Requires sampling for large datasets |

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up your API key

```bash
cp .env.example .env
# Edit .env and add your Google Gemini API key
# Get a free key at: https://aistudio.google.com/apikey
```

The API key is optional — traditional ML features work without it. You only need it for LLM-based analysis (zero-shot classification, aspect extraction, topic extraction).

### 3. Work through the notebook

Open `Project_4_Customer_Review_Analysis.ipynb` in Jupyter and follow the guided walkthrough.

### 4. Run the Streamlit dashboard

```bash
# Test with your completed code
streamlit run review_app.py

# Or test the reference solution
streamlit run solution/review_app.py
```

## Project Structure

```
project-4/
├── data/
│   └── amazon_reviews.csv          # 21K real Amazon reviews
├── solution/
│   ├── review_analyzer.py          # Complete solution — engine
│   └── review_app.py               # Complete solution — Streamlit app
├── review_analyzer.py              # Student version (TODOs 1-10)
├── review_app.py                   # Student version (TODOs 11-15)
├── Project_4_Customer_Review_Analysis.ipynb  # Guided walkthrough
├── requirements.txt                # Python dependencies
├── .env.example                    # API key template
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## TODO Summary

### review_analyzer.py (TODOs 1-10)

| TODO | Function | What to Implement |
|------|----------|-------------------|
| 1 | `load_and_clean()` | Load CSV, drop missing values, parse dates |
| 2 | `clean_text()` | Lowercase, remove punctuation, normalize whitespace |
| 3 | `preprocess_reviews()` | Apply cleaning, add word_count and sentiment_label |
| 4 | `build_tfidf_features()` | Create TfidfVectorizer, fit and transform |
| 5 | `train_sentiment_classifier()` | Train LogisticRegression on TF-IDF features |
| 6 | `classify_sentiment_llm()` | Zero-shot sentiment classification with Gemini |
| 7 | `extract_aspects_llm()` | Extract aspect-level sentiments as JSON |
| 8 | `extract_topics_llm()` | Identify common themes across review samples |
| 9 | `plot_sentiment_distribution()` | Bar chart of sentiment label counts |
| 10 | `ReviewAnalyzer.run_analysis()` | Wire the full ML pipeline together |

### review_app.py (TODOs 11-15)

| TODO | Section | What to Implement |
|------|---------|-------------------|
| 11 | Session State | Initialize engine and analysis_run variables |
| 12 | Run Button | Create ReviewAnalyzer, run analysis, store in state |
| 13 | Overview Tab | Metrics (total reviews, avg rating, etc.) + distribution charts |
| 14 | ML Results Tab | Accuracy display, word importance, custom review classifier |
| 15 | AI Insights Tab | LLM comparison table, topic extraction, aspect analysis |

## GitHub Workflow

### Initial setup

```bash
git init
git add README.md requirements.txt .env.example .gitignore data/ review_analyzer.py review_app.py
git add Project_4_Customer_Review_Analysis.ipynb
git commit -m "Project 4: initial setup with dataset and starter code"
```

### Commit as you go (one commit per part)

```bash
# After completing Part 2 (text preprocessing)
git add review_analyzer.py
git commit -m "feat: implement text cleaning and preprocessing (TODOs 2-3)"

# After completing Part 3 (ML classification)
git add review_analyzer.py
git commit -m "feat: implement TF-IDF features and sentiment classifier (TODOs 4-5)"

# After completing Part 4-5 (LLM analysis)
git add review_analyzer.py
git commit -m "feat: implement LLM classification, aspects, and topics (TODOs 6-8)"

# After completing Part 6 (pipeline + visualization)
git add review_analyzer.py
git commit -m "feat: implement visualization and full pipeline class (TODOs 9-10)"

# After completing the Streamlit app
git add review_app.py
git commit -m "feat: implement Streamlit dashboard (TODOs 11-15)"

# Final commit with notebook
git add Project_4_Customer_Review_Analysis.ipynb
git commit -m "docs: complete notebook with analysis and reflections"
```

### Optional: feature branching

```bash
git checkout -b feature/text-preprocessing
# complete TODOs 2-3
git add review_analyzer.py
git commit -m "feat: text cleaning and preprocessing"
git checkout main
git merge feature/text-preprocessing
```
