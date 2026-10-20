"""Configured JWT/JWKS launch path for the private connector staging service."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from urllib.parse import urlsplit

import jwt
from mcp.server.auth.provider import AccessToken

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.connector_downloads import (
    ConnectorDownloads,
    print_package_reader,
    project_package_reader,
)
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.remote_mcp import create_app
from forge_mind.workspace.store import Store


class JWKSVerifier:
    """Verify signatures and current claims with established PyJWT APIs."""

    def __init__(self, *, issuer: str, jwks_url: str, resource: str) -> None:
        for value in (issuer, jwks_url):
            parts = urlsplit(value)
            if (
                parts.scheme != "https"
                or not parts.hostname
                or parts.username
                or parts.password
                or parts.fragment
            ):
                raise ValueError("Issuer and JWKS configuration require plain HTTPS URLs")
        self.issuer, self.resource = issuer, resource
        self.keys = jwt.PyJWKClient(jwks_url, timeout=10, cache_jwk_set=True, lifespan=300)

    async def verify_token(self, token: str) -> AccessToken | None:
        def verify() -> AccessToken | None:
            try:
                signing_key = self.keys.get_signing_key_from_jwt(token)
                claims = jwt.decode(
                    token,
                    signing_key.key,
                    algorithms=["RS256", "ES256"],
                    audience=self.resource,
                    issuer=self.issuer,
                    options={"require": ["exp", "iss", "sub", "aud"]},
                )
                subject, scope = claims.get("sub"), claims.get("scope", "")
                client = claims.get("client_id") or claims.get("azp")
                if (
                    not isinstance(subject, str)
                    or not subject
                    or not isinstance(scope, str)
                    or not isinstance(client, str)
                    or not client
                ):
                    return None
                return AccessToken(
                    token=token,
                    subject=subject,
                    client_id=client,
                    scopes=scope.split(),
                    expires_at=int(claims["exp"]),
                    resource=self.resource,
                    claims={"iss": self.issuer},
                )
            except (jwt.PyJWTError, ValueError, TypeError, OSError):
                return None

        # Signature checks and fixed-endpoint key fetches do not block the ASGI loop.
        # Every request checks exp/issuer/audience again; only public signing keys cache.
        return await asyncio.to_thread(verify)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--issuer", required=True, help="Configured HTTPS OAuth issuer")
    parser.add_argument("--jwks", required=True, help="Configured HTTPS public signing-key set")
    parser.add_argument(
        "--resource", required=True, help="Exact OAuth resource audience ending /mcp"
    )
    parser.add_argument("--port", type=int, default=4322)
    parser.add_argument(
        "--enable-downloads",
        action="store_true",
        help="Enable short-lived confidential exact-package links at the configured HTTPS resource origin",
    )
    parser.add_argument(
        "--qualify-worker",
        action="store_true",
        help="Exercise exact image/runtime before enabling typed jobs",
    )
    parser.add_argument(
        "--qualification-only",
        action="store_true",
        help="Run worker smoke/control checks and exit without listening",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    verifier = JWKSVerifier(issuer=args.issuer, jwks_url=args.jwks, resource=args.resource)
    executor = Executor("docker")
    if args.qualify_worker or args.qualification_only:
        print(json.dumps(executor.qualify_typed_container(), indent=2))
    if args.qualification_only:
        return
    service = WorkspaceService(Store(args.root), executor)
    downloads = None
    if args.enable_downloads:
        resource_parts = urlsplit(args.resource)
        downloads = ConnectorDownloads(
            origin=f"{resource_parts.scheme}://{resource_parts.netloc}",
            readers={
                "project": project_package_reader(service),
                "print": print_package_reader(service),
            },
        )
    app = create_app(
        service,
        token_verifier=verifier,
        issuer_url=args.issuer,
        resource_url=args.resource,
        downloads=downloads,
    )
    import uvicorn

    # A deployment proxy supplies TLS and the configured public resource audience.
    # This CLI always binds loopback; it never opens a public listener.
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
