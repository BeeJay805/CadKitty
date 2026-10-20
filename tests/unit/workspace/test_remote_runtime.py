from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from forge_mind.workspace.remote_runtime import JWKSVerifier


@pytest.fixture
def signing(monkeypatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    verifier = JWKSVerifier(
        issuer="https://auth.example",
        jwks_url="https://auth.example/jwks",
        resource="https://cad.example/mcp",
    )
    monkeypatch.setattr(
        verifier.keys, "get_signing_key_from_jwt", lambda _: SimpleNamespace(key=key.public_key())
    )
    claims = {
        "iss": verifier.issuer,
        "aud": verifier.resource,
        "sub": "alice",
        "client_id": "chat-client",
        "scope": "cad:read cad:write",
        "exp": int(time.time()) + 300,
    }
    return key, verifier, claims


def test_signed_resource_scoped_token_preserves_exact_owner(signing):
    key, verifier, claims = signing
    token = jwt.encode(claims, key, algorithm="RS256")
    verified = asyncio.run(verifier.verify_token(token))
    assert verified.subject == "alice"
    assert verified.resource == verifier.resource
    assert verified.token == token
    assert verified.claims == {"iss": verifier.issuer}
    assert verified.scopes == ["cad:read", "cad:write"]


@pytest.mark.parametrize(
    "claim,value",
    [
        ("aud", "https://other.example/mcp"),
        ("iss", "https://other.example"),
        ("exp", 1),
        ("scope", ["cad:read"]),
        ("sub", ""),
        ("client_id", ""),
        ("nbf", 9999999999),
    ],
)
def test_signature_does_not_override_invalid_current_claims(signing, claim, value):
    key, verifier, claims = signing
    token = jwt.encode({**claims, claim: value}, key, algorithm="RS256")
    assert asyncio.run(verifier.verify_token(token)) is None


@pytest.mark.parametrize("claim", ["exp", "iss", "aud", "sub", "client_id"])
def test_identity_claims_are_required(signing, claim):
    key, verifier, claims = signing
    claims.pop(claim)
    assert asyncio.run(verifier.verify_token(jwt.encode(claims, key, algorithm="RS256"))) is None


def test_unsigned_and_wrong_signature_tokens_are_refused(signing):
    _, verifier, claims = signing
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    assert asyncio.run(verifier.verify_token(jwt.encode(claims, other, algorithm="RS256"))) is None
    assert asyncio.run(verifier.verify_token(jwt.encode(claims, None, algorithm="none"))) is None


@pytest.mark.parametrize(
    "url",
    [
        "http://auth.example",
        "file:///keys",
        "https://user:secret@auth.example",
        "https://auth.example#fragment",
    ],
)
def test_operator_auth_configuration_requires_https(url):
    with pytest.raises(ValueError, match="HTTPS"):
        JWKSVerifier(
            issuer=url, jwks_url="https://auth.example/jwks", resource="https://cad.example/mcp"
        )
