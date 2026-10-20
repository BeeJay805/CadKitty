"""Explicit OAuth tool policies and structured relinking challenges."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from typing import Any, TypeVar

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.routes import build_resource_metadata_url
from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, InputRequiredResult, TextContent, ToolAnnotations
from pydantic import AnyHttpUrl, ValidationError

Function = TypeVar("Function", bound=Callable[..., Any])


class ConnectorAuthError(ToolError):
    def __init__(self, *, write: bool, insufficient_scope: bool = False) -> None:
        self.scopes = ["cad:read", "cad:write"] if write else ["cad:read"]
        self.code = "insufficient_scope" if insufficient_scope else "invalid_token"
        super().__init__("An authenticated owner with the required CAD scope is required")


def require_owner(
    token: AccessToken | None, *, issuer: str, resource: str, write: bool = False
) -> str:
    """Keep identity checks independent from discovery metadata."""
    if (
        token is None
        or not token.subject
        or (token.claims or {}).get("iss") != issuer
        or token.resource != resource
        or token.expires_at is None
        or token.expires_at <= time.time()
    ):
        raise ConnectorAuthError(write=write)
    if "cad:read" not in token.scopes or (write and "cad:write" not in token.scopes):
        raise ConnectorAuthError(write=write, insufficient_scope=True)
    return token.subject


def auth_challenge(error: ConnectorAuthError, resource: str) -> CallToolResult:
    metadata = build_resource_metadata_url(AnyHttpUrl(resource))
    challenge = (
        f'Bearer resource_metadata="{metadata}", error="{error.code}", '
        f'error_description="Sign in with the required CAD permissions", '
        f'scope="{" ".join(error.scopes)}"'
    )
    return CallToolResult(
        content=[TextContent(type="text", text=str(error))],
        is_error=True,
        _meta={"mcp/www_authenticate": [challenge]},
    )


class ScopedMCPServer(MCPServer[Any]):
    """Keep original tool signatures and results while advertising exact scopes."""

    def __init__(
        self,
        *args: Any,
        issuer: str,
        resource: str,
        access_token_getter: Callable[[], AccessToken | None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.connector_issuer = issuer
        self.connector_resource = resource
        self._connector_access_token_getter = access_token_getter
        self.connector_tool_scopes: dict[str, bool] = {}
        self.connector_file_tools: set[str] = set()
        self.middleware.append(self._scope_descriptors)

    async def _scope_descriptors(
        self, ctx: ServerRequestContext[Any, Any], call_next: CallNext
    ) -> HandlerResult:
        """Add only the host scope extension after the SDK validates its envelope.

        MCP2 sieves core Tool fields. Its public middleware preserves extension
        fields added to the validated wire dictionary on the response path.
        """
        result = await call_next(ctx)
        if ctx.method != "tools/list":
            return result
        if not isinstance(result, dict) or not isinstance(result.get("tools"), list):
            raise ToolError("Tool discovery returned an invalid wire inventory")
        tools = []
        for tool in result["tools"]:
            if not isinstance(tool, dict) or tool.get("name") not in self.connector_tool_scopes:
                raise ToolError("Tool discovery has no registered scope policy")
            write = self.connector_tool_scopes[tool["name"]]
            schemes = [
                {"type": "oauth2", "scopes": ["cad:read", "cad:write"] if write else ["cad:read"]}
            ]
            if (tool.get("_meta") or {}).get("securitySchemes") != schemes:
                raise ToolError("Tool discovery scope policy changed")
            tools.append({**tool, "securitySchemes": schemes})
        return {**result, "tools": tools}

    async def call_tool(
        self, name: str, arguments: dict[str, Any], context: Any = None
    ) -> CallToolResult | InputRequiredResult:
        if name in self.connector_tool_scopes:
            try:
                require_owner(
                    (
                        self._connector_access_token_getter()
                        if self._connector_access_token_getter is not None
                        else get_access_token()
                    ),
                    issuer=self.connector_issuer,
                    resource=self.connector_resource,
                    write=self.connector_tool_scopes[name],
                )
            except ConnectorAuthError as exc:
                return auth_challenge(exc, self.connector_resource)
        try:
            return await super().call_tool(name, arguments, context)
        except ToolError as exc:
            if name not in self.connector_file_tools or not isinstance(
                exc.__cause__, ValidationError
            ):
                raise
            # The outer SDK reports rejected argument values. A file descriptor can
            # contain a confidential URL even in an invalid field or field name.
            # Return before its request handler formats or logs the exception.
            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text="File tool arguments do not match the input schema"
                    )
                ],
                is_error=True,
            )


def scoped_tool(
    server: ScopedMCPServer,
    *,
    annotations: ToolAnnotations,
    file_params: tuple[str, ...] = (),
) -> Callable[[Function], Function]:
    """Register unchanged functions; both descriptor locations carry the same policy."""
    write = annotations.read_only_hint is not True
    schemes = [{"type": "oauth2", "scopes": ["cad:read", "cad:write"] if write else ["cad:read"]}]
    meta: dict[str, Any] = {"securitySchemes": schemes}
    if file_params:
        if (
            not isinstance(file_params, tuple)
            or len(set(file_params)) != len(file_params)
            or any(not isinstance(name, str) or not name.isidentifier() for name in file_params)
        ):
            raise ValueError("File parameters must be unique top-level parameter names")
        meta["openai/fileParams"] = list(file_params)
    register = server.tool(annotations=annotations, meta=meta)

    def decorator(function: Function) -> Function:
        if any(name not in inspect.signature(function).parameters for name in file_params):
            raise ValueError("File parameter does not name a top-level tool argument")
        server.connector_tool_scopes[function.__name__] = write
        if file_params:
            server.connector_file_tools.add(function.__name__)
        return register(function)

    return decorator
