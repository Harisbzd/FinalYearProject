import jwt
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, current_app
import os

# Secret key for JWT (in production, use a strong secret key)
SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')

def generate_token(user_id, email):
    """Generate JWT token for user"""
    payload = {
        'user_id': str(user_id),
        'email': email,
        'exp': datetime.utcnow() + timedelta(hours=24),  # Token expires in 24 hours
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def verify_token(token):
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def token_required(f):
    """Decorator to require valid JWT token"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Get token from header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]  # Bearer <token>
            except IndexError:
                return jsonify({'message': 'Token is missing'}), 401
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            payload = verify_token(token)
            if payload is None:
                return jsonify({'message': 'Token is invalid or expired'}), 401
            
            # Add user info to request context
            request.user_id = payload['user_id']
            request.user_email = payload['email']
            
        except Exception as e:
            return jsonify({'message': 'Token is invalid'}), 401
        
        return f(*args, **kwargs)
    
    return decorated 