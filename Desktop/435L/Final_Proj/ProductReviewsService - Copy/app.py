from flask import Flask
from routes.review_routes import review_routes
from routes.admin_routes import admin_routes
from database import create_tables

def create_app():
    app = Flask(__name__)

    # Register blueprints
    app.register_blueprint(review_routes)
    app.register_blueprint(admin_routes)

    # Setup code that was in before_first_request
    create_tables()  # Ensure tables are created at startup

    return app

# Entry point
if __name__ == '__main__':
    app = create_app()  # Create the Flask app
    app.run(debug=True, port=5000)  # You can adjust the port as necessary
