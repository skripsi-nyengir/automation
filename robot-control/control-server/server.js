const { WebSocketServer } = require('ws');

const PORT = 3000;
const wss = new WebSocketServer({ port: PORT });

let esp32 = null;
const webClients = new Set();

function broadcastEsp32Status() {
  const status = esp32 && esp32.readyState === 1 ? 'connected' : 'disconnected';
  const msg = `esp32status:${status}`;
  webClients.forEach((ws) => {
    if (ws.readyState === 1) ws.send(msg);
  });
}

wss.on('connection', (ws) => {
  let identity = 'unknown';

  ws.on('message', (data) => {
    const msg = data.toString();

    // ESP32 identifies itself on first message
    if (msg === 'ESP32') {
      identity = 'esp32';
      esp32 = ws;
      console.log('[+] ESP32 connected');
      broadcastEsp32Status();
      return;
    }

    // First non-ESP32 message → this is a web client
    if (identity === 'unknown') {
      identity = 'web';
      webClients.add(ws);
      console.log('[+] Web client connected, total:', webClients.size);
      // Tell web client current ESP32 status immediately
      ws.send(
        `esp32status:${esp32 && esp32.readyState === 1 ? 'connected' : 'disconnected'}`
      );
    }

    // Web client → forward to ESP32 + broadcast to other web clients
    if (identity === 'web') {
      if (esp32 && esp32.readyState === 1) {
        esp32.send(msg);
      }
      webClients.forEach((c) => {
        if (c !== ws && c.readyState === 1) c.send(msg);
      });
    }

    // ESP32 → forward to all web clients
    if (identity === 'esp32') {
      webClients.forEach((c) => {
        if (c.readyState === 1) c.send(msg);
      });
    }
  });

  ws.on('close', () => {
    if (identity === 'esp32') {
      esp32 = null;
      console.log('[-] ESP32 disconnected');
      broadcastEsp32Status();
    } else if (identity === 'web') {
      webClients.delete(ws);
      console.log('[-] Web client disconnected, total:', webClients.size);
    }
  });

  ws.on('error', () => {});
});

console.log(`Control Server running on port ${PORT}`);
