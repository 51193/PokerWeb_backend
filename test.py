import jwt
from datetime import datetime, timedelta
SECRET_KEY = 'your_secret_key'
def create_token(user_id):
    payload = {
        'exp': datetime.utcnow() + timedelta(days=1),  # 令牌有效期为1天
        'iat': datetime.utcnow(),
        'sub': user_id  # 通常 'sub' 表示 subject，即用户的唯一标识
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
    return token

def verify_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload['sub']
    except jwt.ExpiredSignatureError:
        return None  # Token has expired
    except jwt.InvalidTokenError:
        return None  # Token is invalid

print(create_token("wer"))

verify_token = verify_token(create_token("qwe"))
print(verify_token)