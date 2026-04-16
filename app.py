"""NLP Finance Sentiment Analysis - Flask Web App (Portfolio Edition)"""
import os
import re
import io
import csv
import json
import time
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, cross_val_predict, learning_curve
from sklearn.preprocessing import label_binarize, LabelEncoder, StandardScaler
from sklearn import metrics
from flask import Flask, render_template, request, jsonify, Response
import feedparser

# Optional: langdetect
try:
    from langdetect import detect as lang_detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# Optional: yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# ---- NLP Setup ----
PUNCTUATION = "!#$%&()*,-./:;<=>?@^_`'{|}~"
PUNCTUATION_RE = re.compile(f"[{PUNCTUATION}]")
STOPWORD_SET = set(stopwords.words('english'))
for word in ('not', 'no', 'nor', 'neither', 'never', 'nobody', 'nothing'):
    STOPWORD_SET.discard(word)

LABELS = ["negative", "neutral", "positive"]
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_FILE = os.path.join(DATA_DIR, "finsentiment.db")

# ---- Negation Handling ----
NEGATION_WORDS = {'not', 'no', 'nor', 'never', 'neither', "n't", 'nobody',
                  'nothing', 'nowhere', 'hardly', 'barely', 'scarcely'}
NEGATION_END = {'.', ',', '!', '?', ';', ':', 'but', 'however', 'though'}


def apply_negation(tokens):
    result = []
    negating = False
    for token in tokens:
        lower = token.lower()
        if lower in NEGATION_WORDS or lower.endswith("n't"):
            negating = True
            result.append(token)
        elif lower in NEGATION_END:
            negating = False
            result.append(token)
        elif negating:
            result.append(f"NOT_{token}")
        else:
            result.append(token)
    return result


# ---- Ticker / Entity Detection ----
KNOWN_COMPANIES = {
    'AAPL': 'Apple', 'GOOG': 'Alphabet', 'GOOGL': 'Alphabet', 'MSFT': 'Microsoft',
    'AMZN': 'Amazon', 'TSLA': 'Tesla', 'META': 'Meta', 'NVDA': 'NVIDIA',
    'JPM': 'JPMorgan', 'BAC': 'Bank of America', 'WMT': 'Walmart', 'DIS': 'Disney',
    'NFLX': 'Netflix', 'INTC': 'Intel', 'AMD': 'AMD', 'CRM': 'Salesforce',
    'V': 'Visa', 'MA': 'Mastercard', 'PFE': 'Pfizer', 'JNJ': 'Johnson & Johnson',
    'XOM': 'ExxonMobil', 'CVX': 'Chevron', 'NKE': 'Nike', 'TGT': 'Target',
    'HD': 'Home Depot', 'BA': 'Boeing', 'GS': 'Goldman Sachs', 'MS': 'Morgan Stanley',
    'UBER': 'Uber', 'LYFT': 'Lyft', 'SNAP': 'Snap', 'TWTR': 'Twitter',
    'PYPL': 'PayPal', 'SQ': 'Block', 'COIN': 'Coinbase', 'HOOD': 'Robinhood',
}
COMPANY_TO_TICKER = {v.lower(): k for k, v in KNOWN_COMPANIES.items()}
TICKER_PATTERN = re.compile(r'\b[A-Z]{1,5}\b')


def detect_entities(text):
    entities = []
    text_lower = text.lower()
    for company, ticker in COMPANY_TO_TICKER.items():
        if company in text_lower:
            entities.append({'ticker': ticker, 'name': KNOWN_COMPANIES[ticker], 'type': 'company'})
    for match in TICKER_PATTERN.finditer(text):
        word = match.group()
        if word in KNOWN_COMPANIES and not any(e['ticker'] == word for e in entities):
            entities.append({'ticker': word, 'name': KNOWN_COMPANIES[word], 'type': 'ticker'})
    return entities


# ---- Twitter / Social Noise Cleaning ----

def clean_text(text):
    """Strip Twitter noise before training/inference."""
    # Remove URLs (http/https/www)
    text = re.sub(r'http\S+|https\S+|www\.\S+', '', text)
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    # Remove RT prefix
    text = re.sub(r'^RT\s+', '', text, flags=re.IGNORECASE)
    # Remove hashtag symbol (keep the word)
    text = re.sub(r'#(\w+)', r'\1', text)
    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---- Loughran-McDonald Financial Lexicon ----

LM_POSITIVE = {
    'exceed', 'exceeded', 'exceeds', 'beat', 'beats', 'surpass', 'surpassed',
    'outperform', 'outperformed', 'upgrade', 'upgraded', 'buy', 'bullish',
    'growth', 'grow', 'grew', 'profit', 'profitable', 'profitability',
    'revenue', 'record', 'strong', 'strength', 'gain', 'gains', 'rise',
    'rose', 'risen', 'rally', 'rallied', 'boom', 'booming', 'opportunity',
    'opportunities', 'benefit', 'benefits', 'positive', 'improve', 'improved',
    'improvement', 'robust', 'solid', 'healthy', 'favorable', 'efficient',
    'innovation', 'innovative', 'expand', 'expansion', 'dividend', 'dividends',
    'acquire', 'acquisition', 'partnership', 'launch', 'launches', 'optimistic',
    'confidence', 'confident', 'success', 'successful', 'advance', 'advances',
    'appreciation', 'recover', 'recovery', 'increase', 'increases', 'increased',
    'accelerate', 'acceleration', 'breakthrough', 'demand', 'win', 'wins', 'won',
}

LM_NEGATIVE = {
    'miss', 'missed', 'misses', 'below', 'disappoint', 'disappointed',
    'disappointing', 'disappointment', 'decline', 'declined', 'declines',
    'fall', 'fell', 'fallen', 'drop', 'dropped', 'drops', 'loss', 'losses',
    'lose', 'losing', 'weak', 'weakness', 'poor', 'worsen', 'worsening',
    'downgrade', 'downgraded', 'sell', 'bearish', 'recession', 'crisis',
    'debt', 'default', 'bankruptcy', 'bankrupt', 'layoff', 'layoffs',
    'restructure', 'restructuring', 'cut', 'cuts', 'reduce', 'reduction',
    'penalty', 'fine', 'lawsuit', 'litigation', 'fraud', 'investigate',
    'investigation', 'probe', 'violation', 'risk', 'risks', 'risky',
    'volatile', 'volatility', 'uncertainty', 'uncertain', 'concern', 'concerns',
    'warning', 'warn', 'warns', 'threat', 'threats', 'negative', 'problem',
    'problems', 'issue', 'issues', 'challenge', 'challenges', 'difficult',
    'difficulty', 'slowdown', 'slump', 'slumped', 'plunge', 'plunged',
    'crash', 'collapse', 'suspend', 'halt', 'terminate', 'writedown', 'impair',
}


def compute_lm_features(texts):
    """
    Compute Loughran-McDonald lexicon features for a list of texts.
    Returns numpy array of shape (n_samples, 3):
      [lm_pos_count, lm_neg_count, lm_sentiment_ratio]
    """
    features = []
    for text in texts:
        words = re.findall(r'\b\w+\b', text.lower())
        pos_count = sum(1 for w in words if w in LM_POSITIVE)
        neg_count = sum(1 for w in words if w in LM_NEGATIVE)
        total = pos_count + neg_count
        ratio = (pos_count - neg_count) / total if total > 0 else 0.0
        features.append([pos_count, neg_count, ratio])
    return np.array(features, dtype=float)


# ---- Text Processing ----

def remove_stopwords(sentence):
    no_punct = PUNCTUATION_RE.sub(" ", sentence)
    words = word_tokenize(no_punct)
    return [w for w in words if w.lower() not in STOPWORD_SET]


def preprocess_sentences(sentences, use_negation=True):
    results = []
    for s in sentences:
        tokens = word_tokenize(s)
        filtered = [w for w in tokens if w.lower() not in STOPWORD_SET]
        filtered = [PUNCTUATION_RE.sub("", w) for w in filtered]
        filtered = [w for w in filtered if w]
        if use_negation:
            filtered = apply_negation(filtered)
        results.append(filtered)
    return results


# ---- Language Detection ----

def detect_language(text):
    if not LANGDETECT_AVAILABLE:
        return 'en'
    try:
        return lang_detect(text)
    except Exception:
        return 'en'


# ---- SQLite Database ----

def init_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            ensemble_label TEXT,
            timestamp TEXT,
            entities TEXT
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            label TEXT,
            timestamp TEXT
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS news_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            link TEXT,
            published TEXT,
            source TEXT,
            sentiment TEXT,
            confidence TEXT,
            entities TEXT,
            fetched_at TEXT
        )''')
        conn.commit()

    # Migrate existing JSON data if present
    _migrate_json_to_db()


def _migrate_json_to_db():
    """One-time migration from JSON flat files to SQLite."""
    feedback_file = os.path.join(DATA_DIR, "feedback.json")
    history_file = os.path.join(DATA_DIR, "history.json")

    if os.path.exists(feedback_file):
        try:
            with open(feedback_file) as f:
                items = json.load(f)
            with sqlite3.connect(DB_FILE) as conn:
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM feedback")
                if c.fetchone()[0] == 0:
                    for item in items:
                        c.execute("INSERT INTO feedback (text, label, timestamp) VALUES (?,?,?)",
                                  (item.get('text', ''), item.get('label', ''),
                                   item.get('timestamp', '')))
                    conn.commit()
            os.rename(feedback_file, feedback_file + '.migrated')
        except Exception as e:
            print(f"Feedback migration error: {e}")

    if os.path.exists(history_file):
        try:
            with open(history_file) as f:
                items = json.load(f)
            with sqlite3.connect(DB_FILE) as conn:
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM history")
                if c.fetchone()[0] == 0:
                    for item in items:
                        c.execute("INSERT INTO history (text, ensemble_label, timestamp, entities) VALUES (?,?,?,?)",
                                  (item.get('text', ''), item.get('ensemble_label', ''),
                                   item.get('timestamp', ''),
                                   json.dumps(item.get('entities', []))))
                    conn.commit()
            os.rename(history_file, history_file + '.migrated')
        except Exception as e:
            print(f"History migration error: {e}")


def db_add_history(text, ensemble_label, entities):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            "INSERT INTO history (text, ensemble_label, timestamp, entities) VALUES (?,?,?,?)",
            (text, ensemble_label, datetime.now().isoformat(), json.dumps(entities))
        )
        conn.commit()


def db_get_history(limit=50):
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def db_get_history_count():
    with sqlite3.connect(DB_FILE) as conn:
        return conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]


def db_add_feedback(text, label):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            "INSERT INTO feedback (text, label, timestamp) VALUES (?,?,?)",
            (text, label, datetime.now().isoformat())
        )
        conn.commit()


def db_get_feedback():
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM feedback ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def db_get_feedback_count():
    with sqlite3.connect(DB_FILE) as conn:
        return conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]


def db_save_news_cache(articles):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("DELETE FROM news_cache")
        for a in articles:
            conn.execute(
                "INSERT INTO news_cache (title,link,published,source,sentiment,confidence,entities,fetched_at) VALUES (?,?,?,?,?,?,?,?)",
                (a['title'], a['link'], a['published'], a['source'],
                 a['sentiment'], json.dumps(a['confidence']),
                 json.dumps(a['entities']), a.get('fetched_at', datetime.now().isoformat()))
            )
        conn.commit()


def db_get_news_cache():
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM news_cache ORDER BY id DESC").fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d['confidence'] = json.loads(d['confidence']) if d['confidence'] else {}
        d['entities'] = json.loads(d['entities']) if d['entities'] else []
        result.append(d)
    return result


# ---- yfinance Stock Data ----

def get_stock_data(tickers, period="1mo"):
    if not YFINANCE_AVAILABLE:
        return {}
    results = {}
    for ticker in tickers[:4]:
        try:
            data = yf.Ticker(ticker)
            hist = data.history(period=period)
            if hist.empty:
                continue
            closes = [round(float(x), 2) for x in hist['Close'].tolist()]
            dates = [str(d.date()) for d in hist.index]
            change_pct = ((closes[-1] - closes[0]) / closes[0] * 100) if len(closes) >= 2 else 0
            results[ticker] = {
                'closes': closes[-30:],
                'dates': dates[-30:],
                'current': closes[-1] if closes else 0,
                'change_pct': round(change_pct, 2),
                'name': KNOWN_COMPANIES.get(ticker, ticker),
            }
        except Exception as e:
            print(f"yfinance {ticker}: {e}")
    return results


# ---- FinBERT ----

_finbert_pipeline = None
_finbert_lock = threading.Lock()
finbert_status = "not_started"   # not_started | loading | ready | failed

FINBERT_LABEL_MAP = {"positive": "positive", "negative": "negative", "neutral": "neutral"}


def _load_finbert_thread():
    global _finbert_pipeline, finbert_status
    finbert_status = "loading"
    try:
        from transformers import pipeline as hf_pipeline
        print("Loading FinBERT (ProsusAI/finbert)...")
        _finbert_pipeline = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            top_k=None,  # return scores for all labels
            device=-1,  # CPU
        )
        finbert_status = "ready"
        print("FinBERT loaded and ready!")
    except Exception as e:
        finbert_status = "failed"
        print(f"FinBERT load failed: {e}")


def start_finbert_loading():
    t = threading.Thread(target=_load_finbert_thread, daemon=True)
    t.start()


def predict_finbert(text):
    if _finbert_pipeline is None:
        return None
    try:
        raw = _finbert_pipeline(text[:512], truncation=True)
        # top_k=None returns [[{label, score}, ...]] for a single input
        result = raw[0] if isinstance(raw[0], list) else raw
        if isinstance(result, dict):
            result = [result]  # single-dict fallback (top_k=1 / old API)
        conf = {FINBERT_LABEL_MAP.get(r['label'].lower(), r['label'].lower()): round(r['score'] * 100, 1)
                for r in result}
        label = max(conf, key=conf.get)
        return {'label': label, 'confidence': conf, 'model_name': 'FinBERT (Transformer)'}
    except Exception as e:
        print(f"FinBERT inference error: {e}")
        return None


# ---- Automated News Pipeline ----

_news_thread_running = False
NEWS_REFRESH_INTERVAL = 1800  # 30 minutes


def _news_pipeline_thread():
    """Background thread: fetch + classify news every 30 minutes."""
    global _news_thread_running
    _news_thread_running = True
    # Initial fetch after 5 seconds (models need to be ready)
    time.sleep(5)
    while True:
        try:
            print("Auto-fetching news...")
            articles = _fetch_raw_news()
            scored = []
            for a in articles:
                # Try to fetch full article body; fall back to title
                body, snippet = _fetch_article_body(a['link'])
                analysis_text = body if body else a['title']
                preds = predict_sentiment(analysis_text, trained_models)
                ents = detect_entities(a['title'])
                scored.append({
                    **a,
                    'snippet': snippet or a['title'],
                    'analyzed_body': bool(body),
                    'sentiment': preds.get('ensemble', {}).get('label', ''),
                    'confidence': preds.get('ensemble', {}).get('confidence', {}),
                    'entities': ents,
                    'fetched_at': datetime.now().isoformat(),
                })
            db_save_news_cache(scored)
            print(f"News cache updated: {len(scored)} articles")
        except Exception as e:
            print(f"News pipeline error: {e}")
        time.sleep(NEWS_REFRESH_INTERVAL)


def start_news_pipeline():
    t = threading.Thread(target=_news_pipeline_thread, daemon=True)
    t.start()


# ---- News Feed ----
NEWS_FEEDS = [
    'https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL,GOOG,MSFT,AMZN,TSLA&region=US&lang=en-US',
    'https://news.google.com/rss/search?q=stock+market+finance&hl=en-US&gl=US&ceid=US:en',
]


def _fetch_article_body(url, timeout=8):
    """
    Attempt to fetch and extract the full article body from a URL.
    Returns (body_text, snippet) or (None, None) on failure.
    """
    try:
        from newspaper import Article
        a = Article(url, request_timeout=timeout)
        a.download()
        a.parse()
        body = a.text.strip()
        if len(body) < 100:
            return None, None
        # Truncate to ~500 words for model input
        words = body.split()
        truncated = ' '.join(words[:500])
        snippet = ' '.join(words[:40]) + ('…' if len(words) > 40 else '')
        return truncated, snippet
    except Exception:
        return None, None


def _fetch_raw_news():
    articles = []
    for url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                articles.append({
                    'title': entry.get('title', ''),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'source': feed.feed.get('title', 'Unknown'),
                })
        except Exception:
            continue
    seen = set()
    unique = []
    for a in articles:
        if a['title'] not in seen and a['title']:
            seen.add(a['title'])
            unique.append(a)
    return unique[:20]


# ---- Data & Models ----

def load_data():
    df_train = pd.read_csv(os.path.join(DATA_DIR, "finance_train.csv"))
    df_test = pd.read_csv(os.path.join(DATA_DIR, "finance_test.csv"))
    # Apply Twitter noise cleaning to Sentence column
    df_train['Sentence'] = df_train['Sentence'].astype(str).apply(clean_text)
    df_test['Sentence'] = df_test['Sentence'].astype(str).apply(clean_text)
    return df_train, df_test


def get_dataset_stats(df_train, df_test):
    label_counts = df_train['Label'].value_counts()
    return {
        "train_size": len(df_train),
        "test_size": len(df_test),
        "label_distribution": {lbl: int(label_counts.get(lbl, 0)) for lbl in LABELS},
    }


def get_top_words(df, label_str, n=15):
    subset = df[df['Label'] == label_str]
    all_words = []
    for sentence in subset['Sentence'].values:
        all_words.extend(remove_stopwords(sentence))
    counter = Counter(w.lower() for w in all_words if len(w) > 2)
    return dict(counter.most_common(n))


# ---- Stacking Ensemble ----

# Globals for stacking meta-learner
meta_clf = None
meta_label_encoder = None
lm_train_features = None
lm_scaler = None


def train_stacking(models, train_text, train_labels, lm_features):
    """
    Train a stacking meta-learner using out-of-fold predictions from base models
    concatenated with LM lexicon features.

    Returns (meta_clf, meta_label_encoder, lm_scaler).
    """
    base_model_keys = ['logistic_regression', 'naive_bayes', 'random_forest', 'svc']
    oof_probas = []

    for key in base_model_keys:
        if key not in models:
            continue
        info = models[key]
        vect_data = info['vectorizer'].transform(train_text)
        # cross_val_predict gives out-of-fold probability predictions
        oof = cross_val_predict(
            info['model'], vect_data, train_labels,
            cv=3, method='predict_proba'
        )
        oof_probas.append(oof)

    # Scale LM features
    scaler = StandardScaler()
    lm_scaled = scaler.fit_transform(lm_features)

    # Concatenate base model OOF probas + LM features as meta-features
    meta_X = np.hstack(oof_probas + [lm_scaled])

    # Encode labels
    le = LabelEncoder()
    le.fit(LABELS)
    meta_y = le.transform(train_labels)

    # Train meta LogisticRegression
    meta_lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    meta_lr.fit(meta_X, meta_y)

    return meta_lr, le, scaler


def predict_stacking(text, models, meta_clf_model, lm_scaler_obj):
    """
    Build meta-features for a single text and run the stacking meta-learner.
    Returns dict with label and confidence.
    """
    if meta_clf_model is None:
        return None

    processed = preprocess_sentences([text])
    processed_text = [" ".join(processed[0])]

    base_model_keys = ['logistic_regression', 'naive_bayes', 'random_forest', 'svc']
    base_probas = []

    for key in base_model_keys:
        if key not in models:
            # Fill with uniform probabilities if model missing
            base_probas.append(np.array([[1.0 / len(LABELS)] * len(LABELS)]))
            continue
        info = models[key]
        vect = info['vectorizer'].transform(processed_text)
        proba = info['model'].predict_proba(vect)
        # Align classes to LABELS order
        classes = list(info['model'].classes_)
        aligned = np.zeros((1, len(LABELS)))
        for i, lbl in enumerate(LABELS):
            if lbl in classes:
                aligned[0, i] = proba[0, classes.index(lbl)]
        base_probas.append(aligned)

    # LM features
    lm_feat = compute_lm_features([text])
    lm_scaled = lm_scaler_obj.transform(lm_feat)

    meta_X = np.hstack(base_probas + [lm_scaled])
    proba = meta_clf_model.predict_proba(meta_X)[0]

    # Map back to LABELS via label encoder stored in meta_label_encoder global
    le = meta_label_encoder
    classes = list(le.classes_)
    conf = {}
    for i, cls in enumerate(classes):
        conf[cls] = round(float(proba[i]) * 100, 1)

    label = max(conf, key=conf.get)
    return {
        'label': label,
        'model_name': 'Stacking Ensemble (LR Meta-Learner)',
        'confidence': conf,
    }


def train_models(df_train):
    global meta_clf, meta_label_encoder, lm_train_features, lm_scaler

    train_sentences = preprocess_sentences(df_train['Sentence'].values)
    train_labels = np.array(df_train['Label'].tolist())
    train_text = [" ".join(t) for t in train_sentences]

    # Improved TF-IDF params
    tfidf = TfidfVectorizer(
        max_features=8000, ngram_range=(1, 3),
        min_df=2, max_df=0.95, sublinear_tf=True,
        analyzer='word'
    )
    train_vect = tfidf.fit_transform(train_text)

    count_vec = CountVectorizer(max_features=5000, ngram_range=(1, 2),
                                min_df=2, max_df=0.95)
    train_count = count_vec.fit_transform(train_text)

    models = {}

    # Improved LR hyperparams
    lr = LogisticRegression(max_iter=1000, C=5.0, solver='saga', class_weight='balanced')
    lr.fit(train_vect, train_labels)
    models['logistic_regression'] = {'model': lr, 'vectorizer': tfidf, 'name': 'Logistic Regression (TF-IDF)'}

    # Improved NB
    nb = MultinomialNB(alpha=0.1)
    nb.fit(train_count, train_labels)
    models['naive_bayes'] = {'model': nb, 'vectorizer': count_vec, 'name': 'Naive Bayes (BoW)'}

    # Improved RF
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=1,
        class_weight='balanced', random_state=42
    )
    rf.fit(train_vect, train_labels)
    models['random_forest'] = {'model': rf, 'vectorizer': tfidf, 'name': 'Random Forest (TF-IDF)'}

    # LinearSVC with calibration (4th classical model)
    svc = LinearSVC(C=0.5, max_iter=2000)
    svc_calibrated = CalibratedClassifierCV(svc, cv=3)
    svc_calibrated.fit(train_vect, train_labels)
    models['svc'] = {'model': svc_calibrated, 'vectorizer': tfidf, 'name': 'Linear SVC (TF-IDF)'}

    cv_scores = {}
    for key, info in models.items():
        vect_data = info['vectorizer'].transform(train_text)
        scores = cross_val_score(info['model'], vect_data, train_labels, cv=5, scoring='accuracy')
        cv_scores[key] = {
            'mean': round(float(scores.mean()) * 100, 1),
            'std': round(float(scores.std()) * 100, 1),
            'folds': [round(float(s) * 100, 1) for s in scores],
        }

    # Train stacking meta-learner
    print("Training stacking meta-learner...")
    lm_features = compute_lm_features(train_text)
    lm_train_features = lm_features
    try:
        meta_clf, meta_label_encoder, lm_scaler = train_stacking(
            models, train_text, train_labels, lm_features
        )
        print("Stacking meta-learner trained!")
    except Exception as e:
        print(f"Stacking training failed: {e}")
        meta_clf = None
        meta_label_encoder = None
        lm_scaler = None

    return models, cv_scores, tfidf, train_vect, train_labels, train_text


def evaluate_model(model_info, df_test):
    test_sentences = preprocess_sentences(df_test['Sentence'].values)
    test_text = [" ".join(t) for t in test_sentences]
    test_labels = np.array(df_test['Label'].tolist())

    test_vect = model_info['vectorizer'].transform(test_text)
    preds = model_info['model'].predict(test_vect)
    acc = metrics.accuracy_score(test_labels, preds)

    cm = metrics.confusion_matrix(test_labels, preds, labels=LABELS)
    report = metrics.classification_report(test_labels, preds, labels=LABELS, output_dict=True)
    return {
        'accuracy': round(acc * 100, 1),
        'confusion_matrix': cm.tolist(),
        'report': {k: {mk: round(mv, 3) if isinstance(mv, float) else mv
                        for mk, mv in v.items()} if isinstance(v, dict) else round(v, 3)
                   for k, v in report.items()},
    }


def compute_benchmarks(models, df_test):
    test_sentences = preprocess_sentences(df_test['Sentence'].values)
    test_text = [" ".join(t) for t in test_sentences]
    test_labels = np.array(df_test['Label'].tolist())
    test_labels_bin = label_binarize(test_labels, classes=LABELS)

    benchmarks = {}
    for key, info in models.items():
        test_vect = info['vectorizer'].transform(test_text)
        if not hasattr(info['model'], 'predict_proba'):
            continue
        probas = info['model'].predict_proba(test_vect)
        classes = list(info['model'].classes_)

        roc_data, pr_data = {}, {}
        for i, label in enumerate(LABELS):
            if label not in classes:
                continue
            class_idx = classes.index(label)
            y_true = test_labels_bin[:, i]
            y_score = probas[:, class_idx]

            fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
            roc_auc = metrics.auc(fpr, tpr)
            step = max(1, len(fpr) // 50)
            roc_data[label] = {
                'fpr': [round(float(x), 4) for x in fpr[::step]],
                'tpr': [round(float(x), 4) for x in tpr[::step]],
                'auc': round(float(roc_auc), 3),
            }

            precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
            pr_auc = metrics.auc(recall, precision)
            step = max(1, len(precision) // 50)
            pr_data[label] = {
                'precision': [round(float(x), 4) for x in precision[::step]],
                'recall': [round(float(x), 4) for x in recall[::step]],
                'auc': round(float(pr_auc), 3),
            }

        benchmarks[key] = {'name': info['name'], 'roc': roc_data, 'pr': pr_data}
    return benchmarks


def compute_learning_curves(models, train_text, train_labels):
    results = {}
    sizes = np.linspace(0.2, 1.0, 5)
    for key, info in models.items():
        vect_data = info['vectorizer'].transform(train_text)
        train_sizes, train_scores, val_scores = learning_curve(
            info['model'], vect_data, train_labels,
            train_sizes=sizes, cv=3, scoring='accuracy'
        )
        results[key] = {
            'name': info['name'],
            'train_sizes': [int(x) for x in train_sizes],
            'train_mean': [round(float(x), 3) for x in train_scores.mean(axis=1)],
            'train_std': [round(float(x), 3) for x in train_scores.std(axis=1)],
            'val_mean': [round(float(x), 3) for x in val_scores.mean(axis=1)],
            'val_std': [round(float(x), 3) for x in val_scores.std(axis=1)],
        }
    return results


def get_feature_importance(models, n=20):
    importance = {}
    for key, info in models.items():
        mdl = info['model']
        vec = info['vectorizer']
        feature_names = vec.get_feature_names_out()

        # CalibratedClassifierCV wraps LinearSVC — access base estimator for coef_
        base_mdl = mdl
        if hasattr(mdl, 'calibrated_classifiers_'):
            # Use the first calibrated classifier's base estimator for coefficients
            try:
                base_mdl = mdl.calibrated_classifiers_[0].estimator
            except Exception:
                base_mdl = mdl

        if hasattr(base_mdl, 'coef_'):
            classes = list(mdl.classes_) if hasattr(mdl, 'classes_') else LABELS
            per_class = {}
            for i, cls in enumerate(classes):
                coefs = base_mdl.coef_[i] if base_mdl.coef_.shape[0] > 1 else base_mdl.coef_[0]
                top_idx = np.argsort(coefs)[-n:][::-1]
                bottom_idx = np.argsort(coefs)[:n]
                per_class[cls] = {
                    'positive_features': [
                        {'word': feature_names[j], 'weight': round(float(coefs[j]), 3)}
                        for j in top_idx
                    ],
                    'negative_features': [
                        {'word': feature_names[j], 'weight': round(float(coefs[j]), 3)}
                        for j in bottom_idx
                    ],
                }
            importance[key] = {'name': info['name'], 'type': 'coefficients', 'classes': per_class}

        elif hasattr(mdl, 'coef_'):
            classes = list(mdl.classes_)
            per_class = {}
            for i, cls in enumerate(classes):
                coefs = mdl.coef_[i]
                top_idx = np.argsort(coefs)[-n:][::-1]
                bottom_idx = np.argsort(coefs)[:n]
                per_class[cls] = {
                    'positive_features': [
                        {'word': feature_names[j], 'weight': round(float(coefs[j]), 3)}
                        for j in top_idx
                    ],
                    'negative_features': [
                        {'word': feature_names[j], 'weight': round(float(coefs[j]), 3)}
                        for j in bottom_idx
                    ],
                }
            importance[key] = {'name': info['name'], 'type': 'coefficients', 'classes': per_class}

        elif hasattr(mdl, 'feature_importances_'):
            imp = mdl.feature_importances_
            top_idx = np.argsort(imp)[-n:][::-1]
            importance[key] = {
                'name': info['name'],
                'type': 'importance',
                'features': [
                    {'word': feature_names[j], 'weight': round(float(imp[j]), 4)}
                    for j in top_idx
                ],
            }

        elif hasattr(mdl, 'feature_log_prob_'):
            classes = list(mdl.classes_)
            per_class = {}
            for i, cls in enumerate(classes):
                log_prob = mdl.feature_log_prob_[i]
                top_idx = np.argsort(log_prob)[-n:][::-1]
                per_class[cls] = {
                    'positive_features': [
                        {'word': feature_names[j], 'weight': round(float(np.exp(log_prob[j])), 4)}
                        for j in top_idx
                    ],
                }
            importance[key] = {'name': info['name'], 'type': 'probability', 'classes': per_class}

    return importance


def explain_prediction(text, models):
    processed = preprocess_sentences([text])
    processed_text = " ".join(processed[0])

    explanations = {}
    for key, info in models.items():
        mdl = info['model']
        vec = info['vectorizer']
        feature_names = list(vec.get_feature_names_out())
        vect = vec.transform([processed_text])

        pred = mdl.predict(vect)[0]
        proba = mdl.predict_proba(vect)[0] if hasattr(mdl, 'predict_proba') else None
        word_contributions = []

        # Try to get coef_ from base estimator (CalibratedClassifierCV case)
        base_mdl = mdl
        if hasattr(mdl, 'calibrated_classifiers_'):
            try:
                base_mdl = mdl.calibrated_classifiers_[0].estimator
            except Exception:
                base_mdl = None

        if base_mdl is not None and hasattr(base_mdl, 'coef_'):
            classes = list(mdl.classes_) if hasattr(mdl, 'classes_') else LABELS
            if pred in classes:
                pred_idx = classes.index(pred)
                coefs = base_mdl.coef_[pred_idx] if base_mdl.coef_.shape[0] > 1 else base_mdl.coef_[0]
                nonzero = vect.nonzero()[1]
                for idx in nonzero:
                    weight = float(coefs[idx]) * float(vect[0, idx])
                    if abs(weight) > 0.01:
                        word_contributions.append({
                            'word': feature_names[idx],
                            'weight': round(weight, 3),
                            'direction': 'supporting' if weight > 0 else 'opposing',
                        })

        elif hasattr(mdl, 'coef_'):
            classes = list(mdl.classes_)
            pred_idx = classes.index(pred)
            coefs = mdl.coef_[pred_idx]
            nonzero = vect.nonzero()[1]
            for idx in nonzero:
                weight = float(coefs[idx]) * float(vect[0, idx])
                if abs(weight) > 0.01:
                    word_contributions.append({
                        'word': feature_names[idx],
                        'weight': round(weight, 3),
                        'direction': 'supporting' if weight > 0 else 'opposing',
                    })

        elif hasattr(mdl, 'feature_importances_'):
            imp = mdl.feature_importances_
            nonzero = vect.nonzero()[1]
            for idx in nonzero:
                weight = float(imp[idx])
                if weight > 0.001:
                    word_contributions.append({
                        'word': feature_names[idx],
                        'weight': round(weight, 4),
                        'direction': 'important',
                    })

        word_contributions.sort(key=lambda x: abs(x['weight']), reverse=True)
        explanations[key] = {
            'model_name': info['name'],
            'prediction': pred,
            'top_words': word_contributions[:10],
            'confidence': {cls: round(float(p) * 100, 1) for cls, p in zip(mdl.classes_, proba)} if proba is not None else {},
        }

    return explanations


def predict_sentiment(text, models):
    processed = preprocess_sentences([text])
    processed_text = [" ".join(processed[0])]

    results = {}
    all_probas = {}
    for key, info in models.items():
        vect = info['vectorizer'].transform(processed_text)
        pred = info['model'].predict(vect)[0]
        proba = info['model'].predict_proba(vect)[0] if hasattr(info['model'], 'predict_proba') else None

        result = {'label': pred, 'model_name': info['name']}
        if proba is not None:
            classes = list(info['model'].classes_)
            result['confidence'] = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, proba)}
            all_probas[key] = {cls: float(p) for cls, p in zip(classes, proba)}
        results[key] = result

    # FinBERT (if loaded)
    fb_result = predict_finbert(text)
    if fb_result:
        results['finbert'] = fb_result
        all_probas['finbert'] = {k: v / 100.0 for k, v in fb_result['confidence'].items()}

    # Stacking ensemble (primary ensemble)
    stacking_result = None
    if meta_clf is not None and lm_scaler is not None:
        try:
            stacking_result = predict_stacking(text, models, meta_clf, lm_scaler)
        except Exception as e:
            print(f"Stacking prediction error: {e}")
            stacking_result = None

    if stacking_result is not None:
        # Stacking is the main ensemble
        results['ensemble'] = {
            'label': stacking_result['label'],
            'model_name': 'Stacking Ensemble (LR Meta-Learner)',
            'confidence': stacking_result['confidence'],
            'finbert_included': 'finbert' in all_probas,
            'stacking': True,
        }
    elif all_probas:
        # Fallback: weighted probability averaging (no stacking available)
        base_weights = {
            'logistic_regression': 0.30,
            'naive_bayes': 0.15,
            'random_forest': 0.25,
            'svc': 0.30,
        }
        if 'finbert' in all_probas:
            base_weights['finbert'] = 0.40
            non_fb_total = sum(v for k, v in base_weights.items() if k != 'finbert')
            scale = 0.60 / non_fb_total if non_fb_total > 0 else 1.0
            for k in list(base_weights.keys()):
                if k != 'finbert':
                    base_weights[k] *= scale

        ensemble_proba = {}
        for lbl in LABELS:
            ensemble_proba[lbl] = sum(
                all_probas.get(k, {}).get(lbl, 0) * base_weights.get(k, 0.25)
                for k in all_probas
            )
        total = sum(ensemble_proba.values())
        if total > 0:
            ensemble_proba = {k: v / total for k, v in ensemble_proba.items()}
        ensemble_label = max(ensemble_proba, key=ensemble_proba.get)
        results['ensemble'] = {
            'label': ensemble_label,
            'model_name': 'Ensemble (Weighted Vote Fallback)',
            'confidence': {k: round(v * 100, 1) for k, v in ensemble_proba.items()},
            'finbert_included': 'finbert' in all_probas,
            'stacking': False,
        }

    return results


# ---- Flask App ----

app = Flask(__name__)

print("Initializing database...")
init_db()

print("Loading data and training models...")
df_train, df_test = load_data()

# Incorporate feedback into training
fb_rows = db_get_feedback()
if fb_rows:
    fb_df = pd.DataFrame(fb_rows)
    if 'text' in fb_df.columns and 'label' in fb_df.columns:
        fb_df = fb_df.rename(columns={'text': 'Sentence', 'label': 'Label'})
        fb_df = fb_df[fb_df['Label'].isin(LABELS)]
        if len(fb_df) > 0:
            # Apply cleaning to feedback data too
            fb_df['Sentence'] = fb_df['Sentence'].astype(str).apply(clean_text)
            df_train = pd.concat([df_train, fb_df[['Sentence', 'Label']]], ignore_index=True)

dataset_stats = get_dataset_stats(df_train, df_test)
trained_models, cv_scores, main_tfidf, train_vect_matrix, train_labels_arr, train_text_list = train_models(df_train)

model_evaluations = {}
for key, info in trained_models.items():
    model_evaluations[key] = evaluate_model(info, df_test)
    model_evaluations[key]['name'] = info['name']
    if key in cv_scores:
        model_evaluations[key]['cv'] = cv_scores[key]

top_words = {lbl: get_top_words(df_train, lbl) for lbl in LABELS}

print("Computing benchmarks...")
benchmark_data = compute_benchmarks(trained_models, df_test)
learning_curve_data = compute_learning_curves(trained_models, train_text_list, train_labels_arr)
feature_importance_data = get_feature_importance(trained_models)

sample_sentences = [
    "Apple stock surges after strong quarterly earnings beat expectations",
    "Oil prices drop sharply amid global recession fears",
    "Federal Reserve holds interest rates steady as expected",
    "Tesla announces massive layoffs following revenue decline",
    "Amazon reports record profits driven by cloud computing growth",
    "Markets remain flat as investors await economic data",
]

print("Models trained and ready! Starting background services...")

# Start background services
start_finbert_loading()
start_news_pipeline()

print("Server ready!")


# ---- API Routes ----

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/stats')
def api_stats():
    return jsonify({
        'dataset': dataset_stats,
        'models': model_evaluations,
        'top_words': top_words,
        'samples': sample_sentences,
        'feedback_count': db_get_feedback_count(),
        'history_count': db_get_history_count(),
        'finbert_status': finbert_status,
        'stacking': meta_clf is not None,
    })


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    results = predict_sentiment(text, trained_models)
    entities = detect_entities(text)
    explanations = explain_prediction(text, trained_models)
    language = detect_language(text)

    db_add_history(text, results.get('ensemble', {}).get('label', ''), entities)

    return jsonify({
        'text': text,
        'predictions': results,
        'entities': entities,
        'explanations': explanations,
        'language': language,
        'finbert_status': finbert_status,
    })


@app.route('/api/batch', methods=['POST'])
def api_batch():
    data = request.get_json()
    texts = data.get('texts', [])
    if not texts:
        return jsonify({'error': 'No texts provided'}), 400
    results = []
    for text in texts[:100]:
        text = text.strip()
        if not text:
            continue
        preds = predict_sentiment(text, trained_models)
        entities = detect_entities(text)
        language = detect_language(text)
        results.append({'text': text, 'predictions': preds, 'entities': entities, 'language': language})
    return jsonify({'results': results})


@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400
    try:
        content = file.read().decode('utf-8')
        reader = csv.DictReader(io.StringIO(content))
        fieldnames = reader.fieldnames or []
        text_col = next(
            (n for n in fieldnames if n.lower() in ('sentence', 'text', 'headline', 'title', 'content')),
            fieldnames[0] if fieldnames else None
        )
        if not text_col:
            return jsonify({'error': 'Could not find text column'}), 400
        results = []
        for row in list(reader)[:100]:
            text = row.get(text_col, '').strip()
            if text:
                preds = predict_sentiment(text, trained_models)
                entities = detect_entities(text)
                results.append({'text': text, 'predictions': preds, 'entities': entities})
        return jsonify({'results': results, 'column_used': text_col})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    data = request.get_json()
    text = data.get('text', '').strip()
    label = data.get('label', '').strip().lower()
    if not text or label not in LABELS:
        return jsonify({'error': 'Invalid text or label'}), 400
    db_add_feedback(text, label)
    return jsonify({'status': 'saved', 'total_feedback': db_get_feedback_count()})


@app.route('/api/retrain', methods=['POST'])
def api_retrain():
    global df_train, trained_models, cv_scores, model_evaluations, dataset_stats, top_words
    global main_tfidf, train_vect_matrix, train_labels_arr, train_text_list
    global benchmark_data, learning_curve_data, feature_importance_data
    global meta_clf, meta_label_encoder, lm_train_features, lm_scaler

    df_base = pd.read_csv(os.path.join(DATA_DIR, "finance_train.csv"))
    df_base['Sentence'] = df_base['Sentence'].astype(str).apply(clean_text)

    fb_rows = db_get_feedback()
    if fb_rows:
        fb_df = pd.DataFrame(fb_rows)
        fb_df = fb_df.rename(columns={'text': 'Sentence', 'label': 'Label'})
        fb_df = fb_df[fb_df['Label'].isin(LABELS)]
        if len(fb_df) > 0:
            fb_df['Sentence'] = fb_df['Sentence'].astype(str).apply(clean_text)
            df_base = pd.concat([df_base, fb_df[['Sentence', 'Label']]], ignore_index=True)

    df_train = df_base
    dataset_stats = get_dataset_stats(df_train, df_test)
    trained_models, cv_scores, main_tfidf, train_vect_matrix, train_labels_arr, train_text_list = train_models(df_train)

    model_evaluations = {}
    for key, info in trained_models.items():
        model_evaluations[key] = evaluate_model(info, df_test)
        model_evaluations[key]['name'] = info['name']
        if key in cv_scores:
            model_evaluations[key]['cv'] = cv_scores[key]

    top_words = {lbl: get_top_words(df_train, lbl) for lbl in LABELS}
    benchmark_data = compute_benchmarks(trained_models, df_test)
    learning_curve_data = compute_learning_curves(trained_models, train_text_list, train_labels_arr)
    feature_importance_data = get_feature_importance(trained_models)

    return jsonify({
        'status': 'retrained',
        'train_size': len(df_train),
        'stacking': meta_clf is not None,
    })


@app.route('/api/export', methods=['POST'])
def api_export():
    data = request.get_json()
    results = data.get('results', [])
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Text', 'Ensemble', 'LR', 'NB', 'RF', 'SVC', 'FinBERT', 'Pos%', 'Neg%', 'Neu%', 'Language', 'Entities', 'Stacking'])
    for r in results:
        preds = r.get('predictions', {})
        ens = preds.get('ensemble', {})
        conf = ens.get('confidence', {})
        entities = ', '.join(e.get('name', e.get('ticker', '')) for e in r.get('entities', []))
        writer.writerow([
            r.get('text', ''), ens.get('label', ''),
            preds.get('logistic_regression', {}).get('label', ''),
            preds.get('naive_bayes', {}).get('label', ''),
            preds.get('random_forest', {}).get('label', ''),
            preds.get('svc', {}).get('label', 'N/A'),
            preds.get('finbert', {}).get('label', 'N/A'),
            conf.get('positive', ''), conf.get('negative', ''), conf.get('neutral', ''),
            r.get('language', 'en'), entities,
            ens.get('stacking', False),
        ])
    return Response(output.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=sentiment_results.csv'})


@app.route('/api/compare', methods=['POST'])
def api_compare():
    data = request.get_json()
    text_a = data.get('text_a', '').strip()
    text_b = data.get('text_b', '').strip()
    if not text_a or not text_b:
        return jsonify({'error': 'Two texts required'}), 400
    return jsonify({
        'a': {
            'text': text_a,
            'predictions': predict_sentiment(text_a, trained_models),
            'entities': detect_entities(text_a),
            'language': detect_language(text_a),
        },
        'b': {
            'text': text_b,
            'predictions': predict_sentiment(text_b, trained_models),
            'entities': detect_entities(text_b),
            'language': detect_language(text_b),
        },
    })


@app.route('/api/history')
def api_history():
    rows = db_get_history(50)
    # Parse entities JSON strings
    history = []
    trend = {'positive': 0, 'negative': 0, 'neutral': 0}
    for row in rows:
        ents = row.get('entities', '[]')
        if isinstance(ents, str):
            try:
                ents = json.loads(ents)
            except Exception:
                ents = []
        history.append({
            'text': row['text'],
            'ensemble_label': row['ensemble_label'],
            'timestamp': row['timestamp'],
            'entities': ents,
        })
        lbl = row.get('ensemble_label', '')
        if lbl in trend:
            trend[lbl] += 1

    return jsonify({
        'history': history,
        'trend': trend,
        'total': db_get_history_count(),
    })


@app.route('/api/news')
def api_news():
    # Return cached news if available, else fetch live
    cached = db_get_news_cache()
    if cached:
        return jsonify({'articles': cached, 'cached': True})
    # Live fetch
    articles = _fetch_raw_news()
    results = []
    for a in articles:
        preds = predict_sentiment(a['title'], trained_models)
        ents = detect_entities(a['title'])
        results.append({
            **a,
            'sentiment': preds.get('ensemble', {}).get('label', ''),
            'confidence': preds.get('ensemble', {}).get('confidence', {}),
            'entities': ents,
            'fetched_at': datetime.now().isoformat(),
        })
    db_save_news_cache(results)
    return jsonify({'articles': results, 'cached': False})


@app.route('/api/stock-data', methods=['POST'])
def api_stock_data():
    data = request.get_json()
    tickers = data.get('tickers', [])
    period = data.get('period', '1mo')
    if not tickers:
        return jsonify({'error': 'No tickers provided'}), 400
    if not YFINANCE_AVAILABLE:
        return jsonify({'error': 'yfinance not available'}), 503
    result = get_stock_data(tickers, period)
    return jsonify({'stocks': result})


@app.route('/api/benchmarks')
def api_benchmarks():
    return jsonify({
        'roc_pr': benchmark_data,
        'learning_curves': learning_curve_data,
    })


@app.route('/api/explainability')
def api_explainability():
    return jsonify(feature_importance_data)


@app.route('/api/finbert-status')
def api_finbert_status():
    return jsonify({'status': finbert_status})


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(debug=debug, host='0.0.0.0', port=port)
