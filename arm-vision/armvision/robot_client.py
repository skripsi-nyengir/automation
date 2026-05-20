"""WebSocket client to the legacy relay. Runs in a background thread with
auto-reconnect. Registers as a 'web client' by sending a non-ESP32 handshake
on connect (see server.js). Tolerates inbound esp32status messages."""
from __future__ import annotations

import threading

import websocket  # from the `websocket-client` package


class RobotClient:
    def __init__(self, url: str, handshake: str = "vision:hello",
                 reconnect_seconds: float = 3.0):
        self._url = url
        self._handshake = handshake
        self._reconnect = reconnect_seconds
        self._app: websocket.WebSocketApp | None = None
        self._thread: threading.Thread | None = None
        self._connected = False
        self._stop = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop:
            self._app = websocket.WebSocketApp(
                self._url,
                on_open=self._on_open,
                on_close=self._on_close,
                on_error=self._on_error,
                on_message=self._on_message,
            )
            # run_forever returns on disconnect; loop to reconnect.
            self._app.run_forever()
            self._connected = False
            if self._stop:
                break
            # Match the legacy web client's 3s retry.
            for _ in range(int(self._reconnect * 10)):
                if self._stop:
                    return
                threading.Event().wait(0.1)

    def _on_open(self, _app) -> None:
        self._connected = True
        try:
            self._app.send(self._handshake)  # register as web client
        except Exception:
            pass

    def _on_close(self, _app, *_args) -> None:
        self._connected = False

    def _on_error(self, _app, _err) -> None:
        self._connected = False

    def _on_message(self, _app, _msg) -> None:
        pass  # esp32status / echoes are ignored

    def send(self, key: str, value) -> None:
        if not self._connected or self._app is None:
            return
        try:
            self._app.send(f"{key}:{value}")
        except Exception:
            self._connected = False

    def close(self) -> None:
        self._stop = True
        if self._app is not None:
            try:
                self._app.close()
            except Exception:
                pass
