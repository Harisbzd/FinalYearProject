import bcrypt

def hash_password(plain_text_password):
    return bcrypt.hashpw(plain_text_password.encode("utf-8"), bcrypt.gensalt())

def check_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password)