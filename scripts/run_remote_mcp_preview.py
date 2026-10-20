"""Run a loopback protocol proof. This does not create an OAuth authorization server.

Set CADKITTY_MCP_PREVIEW_TOKEN to a random secret of at least 32 characters.
Set CADKITTY_MCP_PREVIEW_OWNER to the existing workspace owner to inspect.
The app never logs this secret, calls a model, or qualifies typed Docker execution.
Public deployment must replace this preview verifier with its authorization server.
"""

from __future__ import annotations

import argparse
import hmac
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp.server.auth.provider import AccessToken

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.remote_mcp import create_app
from forge_mind.workspace.store import Store


class PreviewVerifier:
    def __init__(self, secret: str, owner: str, resource: str) -> None:
        self.secret, self.owner, self.resource = secret, owner, resource

    async def verify_token(self, token: str) -> AccessToken | None:
        if not hmac.compare_digest(token, self.secret):
            return None
        return AccessToken(
            token=token,
            subject=self.owner,
            client_id="loopback-protocol-proof",
            resource=self.resource,
            scopes=["cad:read", "cad:write"],
            claims={"iss": self.resource.removesuffix("/mcp")},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Owned workspace store directory")
    parser.add_argument("--port", type=int, default=4322)
    args = parser.parse_args()
    secret, owner = (
        os.environ.get("CADKITTY_MCP_PREVIEW_TOKEN", ""),
        os.environ.get("CADKITTY_MCP_PREVIEW_OWNER", ""),
    )
    if len(secret) < 32 or not owner or not 1 <= args.port <= 65535:
        parser.error(
            "Set a 32-character preview secret, an owner, and a valid port before starting"
        )
    resource = f"http://127.0.0.1:{args.port}/mcp"
    service = WorkspaceService(Store(args.root), Executor("docker"))
    app = create_app(
        service,
        token_verifier=PreviewVerifier(secret, owner, resource),
        issuer_url=resource.removesuffix("/mcp"),
        resource_url=resource,
    )
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
