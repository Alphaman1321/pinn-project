from app import create_app
import unittest

class ReviewServiceTests(unittest.TestCase):
    def setUp(self):
        app = create_app()  # Create the app using the function
        self.client = app.test_client()

    def test_submit_review(self):
        response = self.client.post('/reviews/submit', json={
            'product_id': 1,
            'user_id': 1,
            'rating': 5,
            'comment': 'Great product!'
        })
        self.assertEqual(response.status_code, 201)

    def test_update_review(self):
        response = self.client.put('/reviews/update/1', json={
            'rating': 4,
            'comment': 'Good product.'
        })
        self.assertEqual(response.status_code, 200)

    def test_delete_review(self):
        response = self.client.delete('/reviews/delete/1')
        self.assertEqual(response.status_code, 200)

    def test_get_product_reviews(self):
        response = self.client.get('/reviews/product/1')
        self.assertEqual(response.status_code, 200)

    def test_get_customer_reviews(self):
        response = self.client.get('/reviews/customer/1')
        self.assertEqual(response.status_code, 200)

    def test_moderate_review(self):
        response = self.client.put('/admin/moderate/1?status=approved')
        self.assertEqual(response.status_code, 200)

    def test_get_review_details(self):
        response = self.client.get('/reviews/details/2')
        self.assertEqual(response.status_code, 200)
