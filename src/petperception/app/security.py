"""Password hashing, session tokens and CSRF tokens.

* Passwords: scrypt (stdlib ``hashlib.scrypt``), n=2**14, r=8, p=1, 16-byte
  salt, 32-byte key; stored as ``scrypt$n$r$p$salt_b64$key_b64``.
* Sessions: 32 random bytes (URL-safe) returned once in an HttpOnly cookie;
  only their SHA-256 is stored.
* CSRF: per-session random token; state-changing requests must echo it in
  the ``X-CSRF-Token`` header (the cookie is also ``SameSite=Strict``).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**14, 8, 1


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    key = hashlib.scrypt(password.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32)
    return "scrypt${}${}${}${}${}".format(SCRYPT_N, SCRYPT_R, SCRYPT_P, base64.b64encode(salt).decode(), base64.b64encode(key).decode())


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt_b64, key_b64 = stored.split("$")
        if algo != "scrypt":
            return False
        key = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt_b64), n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(key, base64.b64decode(key_b64))
    except (ValueError, TypeError):
        return False


# A precomputed hash used to equalize timing when the username does not exist.
DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


def new_token() -> str:
    return secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def new_id() -> str:
    return secrets.token_hex(16)
