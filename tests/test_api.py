"""Unit tests for Flask API endpoints."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import json


@pytest.fixture(scope='module')
def client():
    """Create Flask test client with the trained app."""
    import app as flask_app
    flask_app.app.config['TESTING'] = True
    with flask_app.app.test_client() as client:
        yield client


class TestPredictEndpoint:
    def test_predict_positive(self, client):
        res = client.post('/api/predict',
                          data=json.dumps({'text': 'Apple stock surges after record earnings beat'}),
                          content_type='application/json')
        assert res.status_code == 200
        data = res.get_json()
        assert 'predictions' in data
        assert 'ensemble' in data['predictions']
        assert data['predictions']['ensemble']['label'] in ('positive', 'negative', 'neutral')

    def test_predict_negative(self, client):
        res = client.post('/api/predict',
                          data=json.dumps({'text': 'Market crashes as recession fears grow sharply'}),
                          content_type='application/json')
        assert res.status_code == 200
        data = res.get_json()
        assert 'predictions' in data

    def test_predict_returns_entities(self, client):
        res = client.post('/api/predict',
                          data=json.dumps({'text': 'Tesla TSLA stock rises 5%'}),
                          content_type='application/json')
        data = res.get_json()
        assert 'entities' in data
        assert isinstance(data['entities'], list)

    def test_predict_returns_explanations(self, client):
        res = client.post('/api/predict',
                          data=json.dumps({'text': 'Strong quarterly results drive investor confidence'}),
                          content_type='application/json')
        data = res.get_json()
        assert 'explanations' in data

    def test_predict_returns_language(self, client):
        res = client.post('/api/predict',
                          data=json.dumps({'text': 'Strong earnings drive stock higher'}),
                          content_type='application/json')
        data = res.get_json()
        assert 'language' in data

    def test_predict_empty_text(self, client):
        res = client.post('/api/predict',
                          data=json.dumps({'text': ''}),
                          content_type='application/json')
        assert res.status_code == 400

    def test_predict_has_all_models(self, client):
        res = client.post('/api/predict',
                          data=json.dumps({'text': 'Markets closed higher today'}),
                          content_type='application/json')
        data = res.get_json()
        preds = data['predictions']
        assert 'logistic_regression' in preds
        assert 'naive_bayes' in preds
        assert 'random_forest' in preds
        assert 'ensemble' in preds

    def test_predict_confidence_sums_to_100(self, client):
        res = client.post('/api/predict',
                          data=json.dumps({'text': 'Revenue grew significantly this quarter'}),
                          content_type='application/json')
        data = res.get_json()
        conf = data['predictions']['ensemble']['confidence']
        total = sum(conf.values())
        assert abs(total - 100.0) < 1.0


class TestBatchEndpoint:
    def test_batch_multiple_texts(self, client):
        texts = [
            'Apple stock rises after earnings',
            'Market falls amid economic uncertainty',
            'Fed holds rates steady',
        ]
        res = client.post('/api/batch',
                          data=json.dumps({'texts': texts}),
                          content_type='application/json')
        assert res.status_code == 200
        data = res.get_json()
        assert 'results' in data
        assert len(data['results']) == 3

    def test_batch_empty(self, client):
        res = client.post('/api/batch',
                          data=json.dumps({'texts': []}),
                          content_type='application/json')
        assert res.status_code == 400

    def test_batch_each_result_has_predictions(self, client):
        res = client.post('/api/batch',
                          data=json.dumps({'texts': ['Stocks fell sharply']}),
                          content_type='application/json')
        data = res.get_json()
        r = data['results'][0]
        assert 'predictions' in r
        assert 'entities' in r


class TestCompareEndpoint:
    def test_compare_two_texts(self, client):
        res = client.post('/api/compare',
                          data=json.dumps({
                              'text_a': 'Apple beats earnings expectations',
                              'text_b': 'Apple misses earnings expectations',
                          }),
                          content_type='application/json')
        assert res.status_code == 200
        data = res.get_json()
        assert 'a' in data and 'b' in data
        assert 'predictions' in data['a']
        assert 'predictions' in data['b']

    def test_compare_missing_text(self, client):
        res = client.post('/api/compare',
                          data=json.dumps({'text_a': 'Only one text', 'text_b': ''}),
                          content_type='application/json')
        assert res.status_code == 400


class TestStatsEndpoint:
    def test_stats_returns_data(self, client):
        res = client.get('/api/stats')
        assert res.status_code == 200
        data = res.get_json()
        assert 'dataset' in data
        assert 'models' in data
        assert 'top_words' in data

    def test_stats_dataset_has_counts(self, client):
        res = client.get('/api/stats')
        data = res.get_json()
        dataset = data['dataset']
        assert 'train_size' in dataset
        assert 'test_size' in dataset
        assert dataset['train_size'] > 0
        assert dataset['test_size'] > 0

    def test_stats_has_finbert_status(self, client):
        res = client.get('/api/stats')
        data = res.get_json()
        assert 'finbert_status' in data


class TestHistoryEndpoint:
    def test_history_returns_data(self, client):
        # Make a prediction first
        client.post('/api/predict',
                    data=json.dumps({'text': 'Test prediction for history'}),
                    content_type='application/json')
        res = client.get('/api/history')
        assert res.status_code == 200
        data = res.get_json()
        assert 'history' in data
        assert 'trend' in data
        assert 'total' in data

    def test_history_trend_keys(self, client):
        res = client.get('/api/history')
        data = res.get_json()
        trend = data['trend']
        assert 'positive' in trend
        assert 'negative' in trend
        assert 'neutral' in trend


class TestFeedbackEndpoint:
    def test_submit_feedback(self, client):
        # First make a prediction
        client.post('/api/predict',
                    data=json.dumps({'text': 'Great earnings report today'}),
                    content_type='application/json')
        res = client.post('/api/feedback',
                          data=json.dumps({'text': 'Great earnings report today', 'label': 'positive'}),
                          content_type='application/json')
        assert res.status_code == 200
        data = res.get_json()
        assert data['status'] == 'saved'
        assert 'total_feedback' in data

    def test_feedback_invalid_label(self, client):
        res = client.post('/api/feedback',
                          data=json.dumps({'text': 'Some text', 'label': 'invalid_label'}),
                          content_type='application/json')
        assert res.status_code == 400
