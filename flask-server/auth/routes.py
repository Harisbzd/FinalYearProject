from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
from auth.utils import hash_password, check_password
from auth.jwt_utils import generate_token, token_required

auth = Blueprint("auth", __name__)

@auth.route('/register', methods=['POST'])
def register():
    users = current_app.mongo.db["users"]
    data = request.get_json()
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if users.find_one({"email": email}):
        return jsonify({"status": "error", "message": "User already exists"}), 400

    hashed_pw = hash_password(password)

    user_data = {
        "username": username,
        "email": email,
        "password": hashed_pw,
        "registered_at": datetime.utcnow()
    }
    
    result = users.insert_one(user_data)
    
    # Generate token for new user
    token = generate_token(str(result.inserted_id), email)

    return jsonify({
        "status": "success", 
        "message": "User registered",
        "token": token,
        "user": {
            "id": str(result.inserted_id),
            "username": username,
            "email": email
        }
    }), 201

@auth.route('/login', methods=['POST'])
def login():
    users = current_app.mongo.db["users"]
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    user = users.find_one({"email": email})
    if not user or not check_password(password, user["password"]):
        return jsonify({"status": "error", "message": "Invalid credentials"}), 401

    # Generate token for logged in user
    token = generate_token(str(user["_id"]), email)

    return jsonify({
        "status": "success", 
        "message": "Login successful", 
        "token": token,
        "user": {
            "id": str(user["_id"]),
            "username": user["username"],
            "email": user["email"]
        }
    }), 200

@auth.route('/verify-token', methods=['GET'])
@token_required
def verify_token_route():
    """Verify if the current token is valid"""
    return jsonify({
        "status": "success",
        "message": "Token is valid",
        "user": {
            "id": request.user_id,
            "email": request.user_email
        }
    }), 200

@auth.route('/prediction', methods=['GET'])
@token_required
def prediction_page():
    """Protected route for prediction page"""
    return jsonify({
        "status": "success",
        "message": "Access granted to prediction page",
        "user": {
            "id": request.user_id,
            "email": request.user_email
        }
    }), 200