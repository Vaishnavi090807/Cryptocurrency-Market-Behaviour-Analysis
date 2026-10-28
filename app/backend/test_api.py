"""
Automated Validation Suite for Cryptocurrency Case Study Flask API
Course: 23CSE351 - Foundations of Data Science
"""

import unittest
import json
import os
import sys

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import app

class TestCryptoAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_01_health_endpoint(self):
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'healthy')
        self.assertEqual(data['dataset_records'], 109130)
        self.assertEqual(data['distinct_assets'], 71)
        self.assertEqual(data['questions_count'], 15)
        self.assertEqual(data['hypothesis_tests_count'], 7)
        print("PASS: /api/health")

    def test_02_dataset_summary(self):
        response = self.client.get('/api/dataset-summary')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['data']['cleaned_records'], 109130)
        self.assertEqual(data['data']['daily50_panel_days'], 108)
        self.assertEqual(data['hygiene_audit']['duplicate_rows'], 0)
        self.assertIn('Shiba Inu', data['hygiene_audit']['rounded_zero_prices'])
        print("PASS: /api/dataset-summary")

    def test_03_cryptocurrencies_list(self):
        response = self.client.get('/api/cryptocurrencies')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['total_count'], 71)
        # Check Bitcoin is present
        btc = next((a for a in data['data'] if a['name'] == 'Bitcoin'), None)
        self.assertIsNotNone(btc)
        self.assertEqual(btc['symbol'], 'btc')
        self.assertEqual(btc['asset_type'], 'Volatile')
        print("PASS: /api/cryptocurrencies")

    def test_04_historical_trends(self):
        response = self.client.get('/api/historical-trends?asset=Bitcoin&limit=10')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['asset_queried'], 'Bitcoin')
        self.assertGreater(len(data['data']), 0)
        row = data['data'][0]
        self.assertIn('price_usd', row)
        self.assertIn('vol_24h', row)
        self.assertIn('market_cap', row)
        print("PASS: /api/historical-trends")

    def test_05_eda_endpoint(self):
        response = self.client.get('/api/eda')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('distributions', data)
        self.assertIn('correlation_matrix', data)
        print("PASS: /api/eda")

    def test_06_analysis_questions_list(self):
        response = self.client.get('/api/analysis-questions')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['total_questions'], 15)
        print("PASS: /api/analysis-questions")

    def test_07_all_15_analysis_questions_detail(self):
        for q_id in range(1, 16):
            response = self.client.get(f'/api/analysis/{q_id}')
            self.assertEqual(response.status_code, 200, f"Question {q_id} failed with {response.status_code}")
            data = response.get_json()
            self.assertEqual(data['status'], 'success')
            self.assertEqual(data['data']['id'], q_id)
            self.assertIn('title', data['data'])
            self.assertIn('motivation', data['data'])
            self.assertIn('results', data['data'])
            self.assertIn('interpretation', data['data'])
            self.assertIn('conclusion', data['data'])
        print("PASS: All 15 individual analysis question endpoints (/api/analysis/1 through 15)")

    def test_08_hypothesis_tests(self):
        response = self.client.get('/api/hypothesis-tests')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['total_tests'], 7)
        # Verify Test 1 (Weekend volume drop)
        t1 = data['data'][0]
        self.assertEqual(t1['test_id'], 1)
        self.assertEqual(t1['decision'], 'Reject H0')
        self.assertLess(t1['p_value'], 0.001)

        # Verify Test 4 (Bitcoin mean return drift)
        t4 = data['data'][3]
        self.assertEqual(t4['test_id'], 4)
        self.assertEqual(t4['decision'], 'Fail to Reject H0')
        print("PASS: /api/hypothesis-tests")

    def test_09_model_results(self):
        response = self.client.get('/api/model-results')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        perf = data['data']['performance_comparison']
        self.assertGreater(perf['ols_test']['r2'], 0.45)
        self.assertGreater(perf['random_forest_test']['r2'], 0.60)
        self.assertIn('ols_summary', data['data'])
        self.assertIn('rf_feature_importances', data['data'])
        print("PASS: /api/model-results")

    def test_10_predict_volume_endpoint(self):
        payload = {
            "market_cap": 85000000000,
            "chg_24h": 4.2,
            "chg_7d": 8.5,
            "asset_type": "Volatile",
            "is_weekend": False
        }
        response = self.client.post('/api/predict-volume',
                                    data=json.dumps(payload),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        preds = data['data']['predictions']
        self.assertIn('ols_linear_regression', preds)
        self.assertIn('random_forest', preds)
        self.assertGreater(preds['ols_linear_regression']['predicted_volume_usd'], 0)
        print("PASS: /api/predict-volume (POST)")

    def test_11_export_csv_endpoints(self):
        exports = ['daily-panel', 'summary-by-asset', 'hypothesis-tests', 'ml-predictions']
        for exp in exports:
            response = self.client.get(f'/api/export/{exp}')
            self.assertEqual(response.status_code, 200, f"Export {exp} failed")
            self.assertTrue(any(t in response.content_type for t in ['csv', 'excel']))
            self.assertGreater(len(response.data), 100)
        print("PASS: /api/export/<dataset> (Tableau CSV downloads)")

if __name__ == '__main__':
    unittest.main()
