import sqlite3

DATABASE = 'reviews.db'  # Database location

def create_review(product_id, user_id, rating, comment):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO reviews (product_id, user_id, rating, comment)
        VALUES (?, ?, ?, ?)
    """, (product_id, user_id, rating, comment))
    db.commit()
    db.close()

def update_review(review_id, rating, comment):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
        UPDATE reviews
        SET rating = ?, comment = ?, updated_at = CURRENT_TIMESTAMP
        WHERE review_id = ?
    """, (rating, comment, review_id))
    db.commit()
    db.close()

def delete_review(review_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
        DELETE FROM reviews WHERE review_id = ?
    """, (review_id,))
    db.commit()
    db.close()

def get_product_reviews(product_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
        SELECT * FROM reviews WHERE product_id = ?
    """, (product_id,))
    reviews = cursor.fetchall()
    db.close()
    return reviews

def get_customer_reviews(user_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
        SELECT * FROM reviews WHERE user_id = ?
    """, (user_id,))
    reviews = cursor.fetchall()
    db.close()
    return reviews

def get_review_details(review_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""SELECT * FROM reviews WHERE review_id = ?""", (review_id,))
    review = cursor.fetchone()
    db.close()
    
    # Convert tuple to dictionary (assuming columns are review_id, product_id, user_id, rating, comment)
    if review:
        columns = ['review_id', 'product_id', 'user_id', 'rating', 'comment']  # Adjust based on your actual column names
        review_dict = dict(zip(columns, review))
        return review_dict
    return None

def get_db():
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db

def create_tables():
    db = get_db()
    cursor = db.cursor()

    # Create tables for reviews and users (optional, if needed)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            rating INTEGER NOT NULL,
            comment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (product_id) REFERENCES inventory(item_id),
            FOREIGN KEY (user_id) REFERENCES customers(customer_id)
        );
    """)
    db.commit()

    db.close()

