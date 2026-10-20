"""Bounded server-owned image fetch child. No redirects, proxy, auth or URL logging.

Run as a separate Python file with a JSON request on stdin. The parent enforces
the absolute deadline, including DNS. The child emits only bounded image bytes.
"""

from __future__ import annotations

import base64
import http.client
import ipaddress
import json
import re
import socket
import ssl
import sys
from urllib.parse import urlsplit

MAX_BYTES = 4 * 1024 * 1024


class PhotoFetchError(ValueError):
    """Error codes contain no URL, file ID, headers or remote response text."""


def approved_hosts(raw: str) -> tuple[str, ...]:
    if not raw:
        return ()
    hosts = tuple(raw.split(","))
    for host in hosts:
        if (
            host != host.strip().lower()
            or not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?", host)
            or "." not in host
            or ".." in host
        ):
            raise PhotoFetchError("photo_host_configuration_invalid")
        if any(
            not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
            for label in host.split(".")
        ):
            raise PhotoFetchError("photo_host_configuration_invalid")
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            raise PhotoFetchError("photo_host_configuration_invalid")
    return hosts


def target(url: str, hosts: tuple[str, ...]) -> tuple[str, str]:
    try:
        parsed = urlsplit(url)
        host, port = parsed.hostname, parsed.port
    except (ValueError, TypeError):
        raise PhotoFetchError("photo_url_refused") from None
    if (
        not hosts
        or not isinstance(url, str)
        or len(url) > 8192
        or any(ord(c) <= 32 or ord(c) == 127 for c in url)
        or parsed.scheme != "https"
        or host not in hosts
        or port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or "%" in parsed.netloc
        or "\\" in url
    ):
        raise PhotoFetchError("photo_url_refused")
    return host, (parsed.path or "/") + ("?" + parsed.query if parsed.query else "")


def public_addresses(host: str) -> tuple[str, ...]:
    try:
        records = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError:
        raise PhotoFetchError("photo_dns_failed") from None
    addresses = []
    for record in records:
        try:
            address = ipaddress.ip_address(record[4][0])
        except ValueError as exc:
            raise PhotoFetchError("photo_dns_refused") from exc
        if (
            not address.is_global
            or address.is_multicast
            or address.is_reserved
            or any(
                getattr(address, name, None)
                for name in ("ipv4_mapped", "sixtofour", "teredo", "is_site_local")
            )
        ):
            raise PhotoFetchError("photo_dns_refused")
        addresses.append(str(address))
    if not addresses:
        raise PhotoFetchError("photo_dns_failed")
    return tuple(dict.fromkeys(addresses))


class PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connect only to the validated numeric IP, with original hostname TLS checks."""

    def __init__(self, host: str, address: str):
        super().__init__(host, 443, timeout=3, context=ssl.create_default_context())
        self.address = address

    def connect(self) -> None:
        # A numeric address cannot trigger a second attacker-controlled DNS lookup.
        raw = socket.create_connection((self.address, 443), timeout=3)
        try:
            self.sock = ssl.create_default_context().wrap_socket(raw, server_hostname=self.host)
            self.sock.settimeout(5)
        except BaseException:
            raw.close()
            raise


def fetch_image(url: str, hosts: tuple[str, ...]) -> bytes:
    host, path = target(url, hosts)
    addresses = public_addresses(host)
    connection = PinnedHTTPSConnection(host, addresses[0])
    try:
        connection.request(
            "GET",
            path,
            headers={
                "Host": host,
                "Accept": "image/png,image/jpeg,image/webp",
                "Accept-Encoding": "identity",
                "Connection": "close",
            },
        )
        response = connection.getresponse()
        if response.status != 200:
            raise PhotoFetchError("photo_http_refused")
        if response.getheader("Content-Encoding", "identity").lower() not in {"identity", ""}:
            raise PhotoFetchError("photo_encoding_refused")
        declared = response.getheader("Content-Length")
        if declared is not None and (not declared.isdigit() or int(declared) > MAX_BYTES):
            raise PhotoFetchError("photo_too_large")
        chunks = []
        size = 0
        while True:
            chunk = response.read(min(65536, MAX_BYTES + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_BYTES:
                raise PhotoFetchError("photo_too_large")
            chunks.append(chunk)
        if not size or (declared is not None and size != int(declared)):
            raise PhotoFetchError("photo_incomplete")
        return b"".join(chunks)
    except PhotoFetchError:
        raise
    except Exception:
        raise PhotoFetchError("photo_transfer_failed") from None
    finally:
        connection.close()


def main() -> None:
    try:
        request = json.loads(sys.stdin.buffer.read(16385))
        hosts = approved_hosts(request["allowed_hosts"])
        data = fetch_image(request["url"], hosts)
        result = {"status": "ok", "content_base64": base64.b64encode(data).decode("ascii")}
    except Exception:
        result = {"status": "photo_fetch_refused"}
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
