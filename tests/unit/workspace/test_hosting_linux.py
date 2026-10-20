"""Linux syscall/Unix-IPC qualification with fake kernels, never Docker or models.

Run inside an isolated Linux test container as root. Every UID change occurs in
a disposable child. The parent never changes identity. This does not qualify
actual Docker isolation, daemon mounts, provider access, or a public deployment.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import multiprocessing
import os
import socket
import stat
import sys
import tempfile
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not hasattr(socket, "SO_PEERCRED"),
    reason="Actual Linux Unix sockets/openat/peer credentials are required",
)

GATEWAY_UID, OTHER_UID, SHARED_GID = 65534, 65533, 65534
IMAGE, FENCE = "sha256:" + "a" * 64, "b" * 32


class FakeKernel:
    def __init__(self, *, forever=False, cleanup_fails=False):
        self.forever, self.cleanup_fails = forever, cleanup_fails
        self.created, self.removed = [], []

    def identity(self):
        return IMAGE

    def create(self, name, directory, kind, operation_id):
        self.created.append(operation_id)
        return "c" * 64

    def start(self, identity):
        pass

    def poll(self, identity):
        return None if self.forever else 0

    def remove(self, identity):
        if self.cleanup_fails:
            raise OSError("synthetic cleanup uncertainty")
        self.removed.append(identity)


@contextlib.contextmanager
def child(target, *args):
    process = multiprocessing.get_context("fork").Process(target=target, args=args)
    process.start()
    try:
        yield process
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        assert not process.is_alive(), "Owned qualification child could not be reaped"


def rpc_child(connection, path, uid, gid, wire):
    try:
        # Only this disposable process changes real/effective identities.
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)
        assert os.getuid() == os.geteuid() == uid
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
            channel.settimeout(3)
            channel.connect(str(path))
            channel.sendall(wire)
            channel.shutdown(socket.SHUT_WR)
            with channel.makefile("rb") as response:
                data = response.readline(8193)
        assert len(data) <= 8192
        connection.send((True, json.loads(data)))
    except Exception as exc:
        connection.send((False, type(exc).__name__))
    finally:
        connection.close()


def rpc(path, uid=GATEWAY_UID, gid=SHARED_GID, wire=None):
    wire = (
        wire or json.dumps({"version": 1, "fence": FENCE, "command": "identity"}).encode() + b"\n"
    )
    receiving, sending = multiprocessing.get_context("fork").Pipe(duplex=False)
    try:
        with child(rpc_child, sending, path, uid, gid, wire) as process:
            sending.close()
            assert receiving.poll(5), "Owned IPC child exceeded its bounded deadline"
            result = receiving.recv()
            process.join(timeout=1)
            assert process.exitcode == 0
            return result
    finally:
        receiving.close()


def server_child(root, ipc, staging):
    from forge_mind.workspace.hosting_runner import Runner, RunnerPolicy, serve

    assert os.getuid() == os.geteuid() == 0
    runner = Runner(RunnerPolicy(root, staging, IMAGE, FENCE, GATEWAY_UID), FakeKernel())
    serve(runner, ipc / "runner.sock")


@contextlib.contextmanager
def linux_server():
    if os.getuid() != 0 or os.geteuid() != 0:
        pytest.skip(
            "Run this private qualification container with --user 0:0; never change host UID"
        )
    original = (os.getuid(), os.geteuid(), os.getgid(), os.getegid())
    # /tmp avoids pytest's root-only ancestor directory blocking dropped UIDs.
    with tempfile.TemporaryDirectory(prefix="cadkitty-linux-ipc-", dir="/tmp") as temporary:
        base = Path(temporary)
        base.chmod(0o755)
        ipc, staging = base / "ipc", base / "incoming"
        ipc.mkdir()
        os.chown(ipc, 0, SHARED_GID)
        ipc.chmod(0o2770)
        staging.mkdir()
        os.chown(staging, GATEWAY_UID, SHARED_GID)
        staging.chmod(0o2770)
        try:
            with child(server_child, base / "private", ipc, staging) as process:
                path = ipc / "runner.sock"
                deadline = time.monotonic() + 5
                while not path.exists() or stat.S_IMODE(path.stat().st_mode) != 0o660:
                    assert process.is_alive(), "Owned runner exited before socket creation"
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                yield path, ipc, staging
        finally:
            assert original == (os.getuid(), os.geteuid(), os.getgid(), os.getegid())


def uid_guard_child(connection, root):
    from forge_mind.workspace.hosting import HostingError
    from forge_mind.workspace.hosting_runner import Runner, RunnerPolicy, serve

    try:
        staging = root / "incoming"
        staging.mkdir()
        runner = Runner(
            RunnerPolicy(root / "private", staging, IMAGE, FENCE, os.getuid()), FakeKernel()
        )
        with pytest.raises(HostingError, match="distinct dedicated UIDs"):
            serve(runner, root / "must-not-exist.sock")
        assert not (root / "must-not-exist.sock").exists()
        connection.send(True)
    except Exception as exc:
        connection.send(repr(exc))
    finally:
        connection.close()


def test_linux_peer_credentials_accept_gateway_and_reject_another_uid(tmp_path):
    bounded_result(uid_guard_child, tmp_path)
    with linux_server() as (path, ipc, staging):
        assert stat.S_IMODE(path.stat().st_mode) == 0o660
        assert path.stat().st_gid == SHARED_GID
        assert ipc.stat().st_uid == 0 and stat.S_IMODE(ipc.stat().st_mode) == 0o2770
        assert (staging.stat().st_uid, staging.stat().st_gid) == (GATEWAY_UID, SHARED_GID)
        accepted, result = rpc(path)
        assert accepted and result == {
            "version": 1,
            "image": IMAGE,
            "fence": FENCE,
            "restricted": True,
        }
        # Same GID permits connection; SO_PEERCRED must reject the different UID.
        connected, refused = rpc(path, uid=OTHER_UID)
        assert connected and refused["protocol_error"] is True
        assert "peer UID" in refused["error"]


def test_linux_ipc_parent_and_shared_gid_permissions_are_required():
    with linux_server() as (path, ipc, _):
        connected, result = rpc(path, gid=OTHER_UID)
        assert not connected and result == "PermissionError"
        ipc.chmod(0o700)
        try:
            connected, result = rpc(path)
            assert not connected and result == "PermissionError"
        finally:
            ipc.chmod(0o2770)
        assert rpc(path)[0]


def fd_child(connection, root):
    from forge_mind.workspace.hosting import HostingError
    from forge_mind.workspace.hosting_runner import _copy_input, _relative_open

    try:
        source, outside = root / "source", root / "outside"
        source.mkdir()
        outside.mkdir()
        (source / "plain").write_bytes(b"safe")
        (outside / "secret").write_bytes(b"outside")
        (source / "escape").symlink_to(outside, target_is_directory=True)
        (source / "linked").symlink_to(outside / "secret")
        os.link(outside / "secret", source / "hard")
        os.mkfifo(source / "fifo")
        descriptor = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for name in ("escape/secret", "linked", "hard", "../outside/secret"):
                try:
                    opened = _relative_open(descriptor, name, os.O_RDONLY | os.O_NONBLOCK)
                except (HostingError, OSError):
                    continue
                else:
                    os.close(opened)
                    raise AssertionError("Escape unexpectedly opened: " + name)
            with pytest.raises(HostingError, match="special"):
                _copy_input(
                    descriptor,
                    "fifo",
                    {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()},
                    root / "copied",
                )
            # Replace the ticket's path after acquiring its descriptor. The old
            # directory stays anchored; the new symlink cannot redirect openat.
            source.rename(root / "original")
            source.symlink_to(outside, target_is_directory=True)
            opened = _relative_open(descriptor, "plain", os.O_RDONLY)
            with os.fdopen(opened, "rb") as original:
                assert original.read() == b"safe"
        finally:
            os.close(descriptor)
        connection.send(True)
    except Exception as exc:
        connection.send(repr(exc))
    finally:
        connection.close()


def bounded_result(target, *args):
    receiving, sending = multiprocessing.get_context("fork").Pipe(duplex=False)
    try:
        with child(target, sending, *args) as process:
            sending.close()
            assert receiving.poll(8), "Owned syscall child exceeded its bounded deadline"
            result = receiving.recv()
            process.join(timeout=1)
            assert process.exitcode == 0
            assert result is True, result
    finally:
        receiving.close()


def test_linux_openat_rejects_links_fifo_and_path_replacement(tmp_path):
    bounded_result(fd_child, tmp_path)


def stop_child(connection, root):
    from forge_mind.workspace.hosting_backup import inventory
    from forge_mind.workspace.hosting_runner import Runner, RunnerPolicy

    try:
        for index, kind in enumerate(("cancel", "timeout", "cleanup")):
            staging = root / ("incoming-" + kind)
            staging.mkdir()
            ticket = staging / ("d" * 32)
            ticket.mkdir()
            (ticket / "probe.json").write_text("{}")
            backend = FakeKernel(forever=True, cleanup_fails=kind == "cleanup")
            runner = Runner(
                RunnerPolicy(root / ("private-" + kind), staging, IMAGE, FENCE, GATEWAY_UID),
                backend,
            )
            operation_id = f"{index + 1:032x}"
            request = {
                "version": 1,
                "fence": FENCE,
                "command": "launch",
                "operation_id": operation_id,
                "operation_kind": "probe",
                "staging_ticket": ticket.name,
                "inputs": inventory(ticket),
                "remaining_seconds": 0.2 if kind != "cancel" else 2,
            }
            runner.dispatch(request)
            deadline = time.monotonic() + 3
            if kind == "cancel":
                while runner._read(operation_id)["state"] != "running":
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                runner.dispatch(
                    {
                        "version": 1,
                        "fence": FENCE,
                        "command": "cancel",
                        "operation_id": operation_id,
                    }
                )
            while (
                runner._read(operation_id)["state"] not in {"cancelled", "error", "uncertain"}
                or runner._cancel
            ):
                assert time.monotonic() < deadline
                time.sleep(0.01)
            receipt = runner._read(operation_id)
            assert len(backend.created) == 1
            if kind == "cleanup":
                assert receipt["state"] == "uncertain" and backend.removed == []
                assert runner.respond(request).get("preaccept_rejection") is not True
            else:
                assert backend.removed == ["c" * 64]
                assert receipt["state"] == ("cancelled" if kind == "cancel" else "error")
        connection.send(True)
    except Exception as exc:
        connection.send(repr(exc))
    finally:
        connection.close()


def test_linux_fake_kernel_cancel_timeout_and_final_cleanup(tmp_path):
    bounded_result(stop_child, tmp_path)


def test_linux_malformed_and_oversized_wire_is_bounded():
    from forge_mind.workspace.hosting_runner import _WIRE_MAXIMUM

    with linux_server() as (path, _, _):
        for wire in (b"not-json\n", b"[]\n", b"x" * (_WIRE_MAXIMUM + 1)):
            connected, result = rpc(path, wire=wire)
            assert connected and result["protocol_error"] is True
            assert len(result["error"]) <= 300
        assert rpc(path)[1]["restricted"] is True
