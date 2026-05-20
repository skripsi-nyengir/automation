import threading
import time

import pytest

from armvision.robot_client import RobotClient

websockets = pytest.importorskip(
    "websockets", reason="pip install websockets to run robot_client tests")
import asyncio


class MockServer:
    """Minimal WebSocket echo/record server on a background asyncio loop."""
    def __init__(self):
        self.received: list[str] = []
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.port = None
        self._ready = threading.Event()

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())
        self._loop.run_forever()

    async def _serve(self):
        async def handler(ws):
            async for msg in ws:
                self.received.append(msg)
        self._server = await websockets.serve(handler, "localhost", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        self._ready.set()

    def start(self):
        self._thread.start()
        assert self._ready.wait(timeout=5)

    def stop(self):
        self._loop.call_soon_threadsafe(self._loop.stop)


@pytest.fixture
def server():
    s = MockServer()
    s.start()
    yield s
    s.stop()


def wait_for(predicate, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_sends_handshake_on_connect(server):
    client = RobotClient(f"ws://localhost:{server.port}/", handshake="vision:hello")
    client.connect()
    assert wait_for(lambda: "vision:hello" in server.received)
    client.close()


def test_send_formats_key_value(server):
    client = RobotClient(f"ws://localhost:{server.port}/", handshake="vision:hello")
    client.connect()
    assert wait_for(lambda: client.is_connected)
    client.send("base", 120)
    assert wait_for(lambda: "base:120" in server.received)
    client.close()


def test_send_before_connect_is_dropped_not_crash():
    client = RobotClient("ws://localhost:59999/", handshake="vision:hello")
    client.send("base", 90)  # not connected -> no exception
    assert client.is_connected is False
