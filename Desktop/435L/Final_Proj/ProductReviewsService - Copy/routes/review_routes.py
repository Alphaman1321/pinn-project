from flask import Blueprint, request, jsonify
from database import create_review, update_review, delete_review, get_product_reviews, get_customer_reviews, get_review_details

review_routes = Blueprint('review_routes', __name__)

@review_routes.route('/reviews/submit', methods=['POST'])
def submit_review():
    data = request.get_json()
    product_id = data.get('product_id')
    user_id = data.get('user_id')
    rating = data.get('rating')
    comment = data.get('comment')

    if not all([product_id, user_id, rating]):
        return jsonify({'error': 'Missing required fields'}), 400

    create_review(product_id, user_id, rating, comment)
    return jsonify({'message': 'Review submitted successfully'}), 201

@review_routes.route('/reviews/update/<int:review_id>', methods=['PUT'])
def update_review_route(review_id):
    data = request.get_json()
    rating = data.get('rating')
    comment = data.get('comment')

    if not any([rating, comment]):
        return jsonify({'error': 'No fields to update'}), 400

    update_review(review_id, rating, comment)
    return jsonify({'message': 'Review updated successfully'}), 200

@review_routes.route('/reviews/delete/<int:review_id>', methods=['DELETE'])
def delete_review_route(review_id):
    delete_review(review_id)
    return jsonify({'message': 'Review deleted successfully'}), 200

@review_routes.route('/reviews/product/<int:product_id>', methods=['GET'])
def get_reviews_for_product(product_id):
    reviews = get_product_reviews(product_id)
    return jsonify([dict(review) for review in reviews]), 200

@review_routes.route('/reviews/customer/<int:user_id>', methods=['GET'])
def get_reviews_for_customer(user_id):
    reviews = get_customer_reviews(user_id)
    return jsonify([dict(review) for review in reviews]), 200

@review_routes.route('/reviews/details/<int:review_id>', methods=['GET'])
def get_review_details_route(review_id):
    review = get_review_details(review_id)
    if review:
        return jsonify(review), 200
    return jsonify({'error': 'Review not found'}), 404

