from flask import Blueprint, request, jsonify
from database import update_review

admin_routes = Blueprint('admin_routes', __name__)

@admin_routes.route('/admin/moderate/<int:review_id>', methods=['PUT'])
def moderate_review(review_id):
    status = request.args.get('status')
    if status not in ['approved', 'rejected']:
        return jsonify({'error': 'Invalid status'}), 400

    # You can implement moderation logic here
    update_review(review_id, None, f"Review {status}")
    return jsonify({'message': f'Review {status} successfully'}), 200
