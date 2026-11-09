"""Threaded TCP connections.

The game's main loop must never block on the network, so every socket gets a
reader thread that decodes frames and drops them on a queue. The main loop
drains the queue once a frame with `poll()`. Sends are short and serialised
with a lock; TCP_NODELAY stops Nagle's algorithm sitting on small state
messages.
"""

from __future__ import annotations

import queue
import socket
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from .protocol import FrameDecoder, ProtocolError, encode


class Connection:
    """One TCP peer. Incoming messages arrive on `inbox` as (message) dicts;
    a None on the inbox means the connection closed."""

    def __init__(self, sock: socket.socket, label: str = "") -> None:
        self.sock = sock
        self.label = label
        self.inbox: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        self._send_lock = threading.Lock()
        self._closed = threading.Event()
        self.close_reason = ""
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._reader = threading.Thread(target=self._read_loop, daemon=True,
                                        name=f"net-read-{label}")
        self._reader.start()

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def send(self, message: Dict[str, Any]) -> bool:
        if self.closed:
            return False
        try:
            data = encode(message)
        except (ProtocolError, ValueError, TypeError):
            # Unencodable (a NaN from a physics blow-up, say). Dropping one
            # state message is harmless; raising here would take the game down.
            return False
        try:
            with self._send_lock:
                self.sock.sendall(data)
            return True
        except OSError as error:
            self.close(f"send failed: {error}")
            return False

    def poll(self) -> List[Optional[Dict[str, Any]]]:
        out = []
        while True:
            try:
                out.append(self.inbox.get_nowait())
            except queue.Empty:
                return out

    def close(self, reason: str = "closed") -> None:
        if self._closed.is_set():
            return
        self.close_reason = self.close_reason or reason
        self._closed.set()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
        self.inbox.put(None)

    def _read_loop(self) -> None:
        decoder = FrameDecoder()
        try:
            while not self._closed.is_set():
                data = self.sock.recv(65536)
                if not data:
                    self.close("peer disconnected")
                    return
                for message in decoder.feed(data):
                    self.inbox.put(message)
        except ProtocolError as error:
            self.close(f"protocol error: {error}")
        except OSError as error:
            self.close(f"connection lost: {error}")


class Listener:
    """Accepts connections on a port and hands each to `on_accept`."""

    def __init__(self, port: int, on_accept: Callable[[Connection, Tuple[str, int]], None],
                 host: str = "0.0.0.0") -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self._on_accept = on_accept
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._accept_loop, daemon=True,
                                        name="net-accept")
        self._thread.start()

    def _accept_loop(self) -> None:
        self.sock.settimeout(0.25)
        counter = 0
        while not self._stop.is_set():
            try:
                client, address = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            client.settimeout(None)
            counter += 1
            self._on_accept(Connection(client, f"peer{counter}"), address)

    def close(self) -> None:
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass


def connect(address: str, port: int, timeout: float = 5.0) -> Connection:
    sock = socket.create_connection((address, port), timeout=timeout)
    sock.settimeout(None)
    return Connection(sock, f"{address}:{port}")


def lan_addresses() -> List[str]:
    """This machine's LAN address(es), to show on the host screen.

    Connecting a UDP socket sends nothing; it just asks the OS which local
    interface would route to that address.
    """
    found = []
    for probe in ("10.255.255.255", "192.168.255.255", "8.8.8.8"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((probe, 1))
                address = s.getsockname()[0]
                if address not in found and not address.startswith("127."):
                    found.append(address)
        except OSError:
            continue
    return found or ["127.0.0.1"]
