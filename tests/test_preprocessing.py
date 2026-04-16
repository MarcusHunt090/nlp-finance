"""Unit tests for text preprocessing pipeline."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from app import apply_negation, preprocess_sentences, detect_entities, detect_language, KNOWN_COMPANIES


class TestNegation:
    def test_basic_negation(self):
        tokens = ['not', 'good', 'earnings']
        result = apply_negation(tokens)
        assert result == ['not', 'NOT_good', 'NOT_earnings']

    def test_negation_ends_at_clause_boundary(self):
        tokens = ['not', 'good', ',', 'still', 'rising']
        result = apply_negation(tokens)
        assert result[1] == 'NOT_good'
        assert result[3] == 'still'  # negation ended at comma
        assert result[4] == 'rising'

    def test_no_negation_without_trigger(self):
        tokens = ['strong', 'earnings', 'growth']
        result = apply_negation(tokens)
        assert result == tokens

    def test_apostrophe_negation(self):
        tokens = ["didn", "n't", 'meet', 'expectations']
        result = apply_negation(tokens)
        assert 'NOT_meet' in result

    def test_never(self):
        tokens = ['never', 'profitable']
        result = apply_negation(tokens)
        assert 'NOT_profitable' in result

    def test_but_ends_negation(self):
        tokens = ['not', 'bad', 'but', 'could', 'improve']
        result = apply_negation(tokens)
        assert result[1] == 'NOT_bad'
        assert result[3] == 'could'


class TestPreprocess:
    def test_basic_preprocessing(self):
        sentences = ["Apple stock surged today"]
        result = preprocess_sentences(sentences)
        assert isinstance(result, list)
        assert len(result) == 1
        tokens = result[0]
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_stopword_removal(self):
        sentences = ["the stock is going up with great momentum"]
        result = preprocess_sentences(sentences)
        tokens = [t.lower() for t in result[0]]
        assert 'the' not in tokens
        assert 'with' not in tokens

    def test_negation_preserved(self):
        sentences = ["not profitable this quarter"]
        result = preprocess_sentences(sentences, use_negation=True)
        joined = ' '.join(result[0]).lower()
        assert 'not' in joined or 'NOT_' in ' '.join(result[0])

    def test_empty_string(self):
        result = preprocess_sentences([""])
        assert result == [[]]

    def test_multiple_sentences(self):
        sentences = ["Stocks rose sharply", "Market fell today"]
        result = preprocess_sentences(sentences)
        assert len(result) == 2


class TestEntityDetection:
    def test_ticker_detection(self):
        entities = detect_entities("AAPL stock rose 3% today")
        tickers = [e['ticker'] for e in entities]
        assert 'AAPL' in tickers

    def test_company_name_detection(self):
        entities = detect_entities("Apple reported strong quarterly earnings")
        tickers = [e['ticker'] for e in entities]
        assert 'AAPL' in tickers

    def test_multiple_entities(self):
        entities = detect_entities("Microsoft and Tesla both beat expectations")
        tickers = [e['ticker'] for e in entities]
        assert 'MSFT' in tickers
        assert 'TSLA' in tickers

    def test_no_entities(self):
        entities = detect_entities("The market was flat today")
        assert isinstance(entities, list)

    def test_no_duplicate_entities(self):
        # "Apple" and "AAPL" in same text shouldn't produce duplicates
        entities = detect_entities("Apple (AAPL) beat earnings")
        aapl_count = sum(1 for e in entities if e['ticker'] == 'AAPL')
        assert aapl_count == 1

    def test_entity_structure(self):
        entities = detect_entities("TSLA surged 10%")
        assert len(entities) >= 1
        assert 'ticker' in entities[0]
        assert 'name' in entities[0]
        assert 'type' in entities[0]


class TestLanguageDetection:
    def test_english_detected(self):
        lang = detect_language("Apple stock rises after strong earnings report")
        assert lang == 'en'

    def test_returns_string(self):
        lang = detect_language("some text here")
        assert isinstance(lang, str)
        assert len(lang) >= 2

    def test_empty_string_fallback(self):
        # Should not crash on empty/short input
        lang = detect_language("")
        assert isinstance(lang, str)
