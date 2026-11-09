import base64, hashlib, hmac, json, os, secrets, time
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from .database import get_db
from .models import User

SECRET = os.getenv("SECRET_KEY", "local-development-only-change-me")
bearer = HTTPBearer(auto_error=False)
password_hasher = PasswordHasher()

def _b64(data: bytes) -> str: return base64.urlsafe_b64encode(data).decode().rstrip("=")
def hash_password(password: str) -> str:
    return password_hasher.hash(password)
def verify_password(password: str, encoded: str) -> bool:
    try:
        return password_hasher.verify(encoded,password)
    except (InvalidHashError,VerifyMismatchError): return False
def issue_token(user_id: int) -> str:
    body = _b64(json.dumps({"sub": user_id, "exp": int(time.time())+900}).encode())
    sig = _b64(hmac.new(SECRET.encode(), body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"
def issue_refresh_token() -> str: return secrets.token_urlsafe(48)
def refresh_hash(token: str) -> str: return hashlib.sha256(token.encode()).hexdigest()
def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer), db: Session = Depends(get_db)) -> User:
    if not credentials: raise HTTPException(401, "Authentication required")
    try:
        body, sig = credentials.credentials.split(".")
        valid = hmac.compare_digest(_b64(hmac.new(SECRET.encode(), body.encode(), hashlib.sha256).digest()), sig)
        data = json.loads(base64.urlsafe_b64decode(body + "=="))
        if not valid or data["exp"] < time.time(): raise ValueError()
        user = db.get(User, int(data["sub"]))
    except Exception: user = None
    if not user: raise HTTPException(401, "Invalid or expired token")
    return user
