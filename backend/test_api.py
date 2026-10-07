import unittest
import json
import os
import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import app

class TestHousePriceAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'online')
        self.assertEqual(data['version'], '2.0.0')

    def test_health_route(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['model_loaded'])
        self.assertEqual(data['status'], 'healthy')

    def test_predict_success_with_uncertainty_and_insights(self):
        payload = {
            "Area(sqft)": 2200,
            "Rooms": 3,
            "Bathrooms": 2,
            "Floors": 2,
            "Location": "Suburban",
            "YearBuilt": 2012,
            "Parking": 2,
            "Condition": "Good"
        }
        response = self.app.post('/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('predicted_price', data)
        self.assertGreater(data['predicted_price'], 50000)
        
        # Test quantile uncertainty interval
        range_data = data['estimated_range']
        self.assertIn('lower_bound', range_data)
        self.assertIn('upper_bound', range_data)
        self.assertLess(range_data['lower_bound'], data['predicted_price'])
        self.assertGreater(range_data['upper_bound'], data['predicted_price'])
        
        # Test feature contributions
        self.assertIn('feature_contributions', data)
        self.assertIn('breakdown', data['feature_contributions'])
        
        # Test comparables
        self.assertIn('comparables', data)
        self.assertEqual(len(data['comparables']), 5)

    def test_predict_out_of_distribution_warning(self):
        # Area = 6,500 sqft exceeds training maximum (5,200 sqft)
        payload = {
            "Area(sqft)": 6500,
            "Rooms": 5,
            "Bathrooms": 4,
            "Floors": 2,
            "Location": "Suburban",
            "YearBuilt": 2015,
            "Parking": 2,
            "Condition": "Good"
        }
        response = self.app.post('/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['out_of_distribution']['is_out_of_distribution'])
        self.assertIn('Area(sqft)', data['out_of_distribution']['flags'])

    def test_predict_negative_area_rejected(self):
        payload = {
            "Area(sqft)": -500,
            "Rooms": 3,
            "Bathrooms": 2,
            "Floors": 1,
            "Location": "Urban",
            "YearBuilt": 2010,
            "Parking": 1,
            "Condition": "Good"
        }
        response = self.app.post('/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("Area(sqft) must be positive", data['message'])

    def test_predict_unknown_location_rejected(self):
        payload = {
            "Area(sqft)": 2000,
            "Rooms": 3,
            "Bathrooms": 2,
            "Floors": 1,
            "Location": "Beach",  # Unknown location!
            "YearBuilt": 2010,
            "Parking": 1,
            "Condition": "Good"
        }
        response = self.app.post('/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("Invalid Location 'Beach'", data['message'])

    def test_predict_missing_feature(self):
        payload = {
            "Area(sqft)": 2500,
            "Rooms": 3
        }
        response = self.app.post('/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("Missing required features", data['message'])

    def test_batch_predict(self):
        payload = [
            {
                "Area(sqft)": 1800,
                "Rooms": 3,
                "Bathrooms": 2,
                "Floors": 1,
                "Location": "Rural",
                "YearBuilt": 1995,
                "Parking": 1,
                "Condition": "Fair"
            },
            {
                "Area(sqft)": 3200,
                "Rooms": 4,
                "Bathrooms": 3,
                "Floors": 2,
                "Location": "Downtown",
                "YearBuilt": 2020,
                "Parking": 2,
                "Condition": "Excellent"
            }
        ]
        response = self.app.post('/predict/batch', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['successful'], 2)
        self.assertEqual(len(data['results']), 2)

    def test_predict_sensitivity_curve(self):
        payload = {
            "Area(sqft)": 2000,
            "Rooms": 3,
            "Bathrooms": 2,
            "Floors": 2,
            "Location": "Urban",
            "YearBuilt": 2015,
            "Parking": 1,
            "Condition": "Good"
        }
        response = self.app.post('/predict/curve', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertGreater(len(data['points']), 10)
        # Verify points increase with area
        prices = [pt['predicted_price'] for pt in data['points']]
        self.assertGreater(prices[-1], prices[0])

if __name__ == '__main__':
    unittest.main()
