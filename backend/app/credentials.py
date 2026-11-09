"""Encryption helpers for provider secrets stored in the database."""
from __future__ import annotations

import base64
import hashlib
import os

from cryptography.fernet import Fernet, InvalidToken


def cipher() -> Fernet:
    configured = os.getenv("CREDENTIAL_ENCRYPTION_KEY")
    if configured:
        try:
            return Fernet(configured.encode())
        except Exception as exc:
            raise RuntimeError("CREDENTIAL_ENCRYPTION_KEY must be a valid Fernet URL-safe base64 key") from exc
    secret = os.getenv("SECRET_KEY", "local-development-only-change-me")
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256((secret + ":raglab-provider-credentials").encode()).digest()))


def encrypt_secret(value: str) -> str:
    return cipher().encrypt(value.encode()).decode()


def decrypt_secret(value: str) -> str:
    try:
        return cipher().decrypt(value.encode()).decode()
    except InvalidToken as exc:
        raise RuntimeError("Provider key could not be decrypted; verify CREDENTIAL_ENCRYPTION_KEY is unchanged.") from exc
