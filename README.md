# FinSentiment — Financial Article Sentiment Analysis

> Deep NLP analysis for financial articles — 5-model ensemble, sentence-level breakdown, topic detection, and FinBERT transformer integration.

## Live Demo

[🚀 Live Demo](https://nlpfinance.up.railway.app)

## What It Does

Paste any financial article URL or text for instant deep analysis:

- **5-Model Ensemble** — Logistic Regression, Naive Bayes, Random Forest, Linear SVC, and a stacking meta-learner
- **FinBERT Integration** — ProsusAI/finbert transformer for state-of-the-art financial NLP
- **Sentence-Level Breakdown** — Every sentence classified and color-coded
- **Sentiment Arc** — Bar chart showing how sentiment shifts through an article
- **Topic Detection** — Auto-tags Earnings, Revenue, Layoffs, Acquisitions, IPO, and more
- **Live News Feed** — Auto-fetches and classifies financial articles every 30 minutes
- **Batch Analysis** — Upload CSV or paste multiple articles at once
- **Word Explainability** — See which words drive the prediction

## Model Performance

| Model | Test Accuracy |
|---|---|
| Stacking Ensemble (LR Meta-Learner) | Best overall |
| Linear SVC | 80.8% |
| Logistic Regression | 79.3% |
| Random Forest | 80.1% |
| Naive Bayes | 79.0% |
| FinBERT (Transformer) | State-of-the-art |

Training data: **9,772 labeled financial sentences** (Malo et al., 2014 + augmented)

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, Flask |
| ML | scikit-learn, PyTorch |
| Transformer | HuggingFace Transformers, ProsusAI/finbert |
| NLP | NLTK, Loughran-McDonald Lexicon |
| Article Parsing | newspaper3k |
| Stock Data | yfinance |
| Database | SQLite |
| Frontend | Vanilla JS, Chart.js |
| Deployment | Docker, Railway |

## Architecture

```
Article URL / Text Input
        ↓
  Text Cleaning & Preprocessing (NLTK)
        ↓
  ┌─────────────────────────────────────┐
  │         5-Model Ensemble             │
  │  LR · NB · RF · SVC · FinBERT      │
  │     + LM Lexicon Features           │
  │   → Stacking Meta-Learner (LR)     │
  └─────────────────────────────────────┘
        ↓
  Sentence Analysis · Topic Detection · Sentiment Arc
        ↓
  Results + Word Contributions + Stock Correlation
```

## Setup

```bash
git clone https://github.com/MarcusHunt090/nlp-finance.git
cd nlp-finance
pip install -r requirements.txt
python app.py
```

Open http://localhost:5001

## Docker

```bash
docker-compose up --build
```

## Features Walkthrough

- **Article URL tab** — Paste a Reuters, Bloomberg, CNBC, or MarketWatch URL
- **Benchmarks** — Full ROC curves, precision-recall, confusion matrices
- **Explainability** — Top positive/negative words per model
- **History** — SQLite-backed analysis history with re-run capability
- **Feedback** — Label corrections feed into model retraining

## Dataset

Trained on the [Financial PhraseBank](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) (Malo et al., 2014) with additional Twitter financial sentiment data, cleaned and augmented to 9,772 training samples across 3 classes: positive, neutral, negative.

---

Built as a portfolio project demonstrating end-to-end NLP system design, from data preprocessing through production deployment.
