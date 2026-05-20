/*
 * Robot Arm - ESP32 Firmware (WebSocket)
 *
 * Hardware:
 *   ESP32 (any dev board)
 *   4x Servo motors (SG90 / MG90S / MG996R)
 *
 * Pin Mapping (change to match your wiring):
 *   GPIO 13 - Base Rotation
 *   GPIO 12 - Arm Up/Down
 *   GPIO 14 - Arm Forward/Back
 *   GPIO 27 - Gripper
 *
 * Required Libraries:
 *   - WebSockets by Markus Sattler (Links2004)
 *   - ESP32Servo by Kevin Harrington, John K. Bennett
 *
 * Message Format (from Web UI):
 *   base:90         - Set Base to 90°
 *   updown:45       - Set Up/Down to 45°
 *   arm:120         - Set Arm to 120°
 *   gripper:60      - Set Gripper to 60°
 *   emergency:STOP  - Emergency stop
 *   emergency:RESUME - Resume from emergency
 *
 * Connection:
 *   ESP32 sends "ESP32" as first message to identify itself
 *   Then listens for commands from Control Server
 */

#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ESP32Servo.h>

// ========== WiFi Config ==========
const char* WIFI_SSID     = "DIRECT-19879533";
const char* WIFI_PASSWORD = "tinggalmasukaja";

// ========== Control Server ==========
const char* WS_HOST       = "10.69.92.67";
const int   WS_PORT       = 3000;
const char* WS_PATH       = "/";

// ========== Pin Mapping ==========
const int PIN_BASE    = 13;
const int PIN_UPDOWN  = 12;
const int PIN_ARM     = 14;
const int PIN_GRIPPER = 27;

// ========== Servo Objects ==========
Servo servoBase;
Servo servoUpdown;
Servo servoArm;
Servo servoGripper;

// ========== WebSocket Client ==========
WebSocketsClient webSocket;

// ========== State ==========
bool emergencyStop = false;
bool wsConnected = false;

// ========== Function Prototypes ==========
void connectWiFi();
void webSocketEvent(WStype_t type, uint8_t* payload, size_t length);
void handleCommand(const char* msg);
void setServoAngle(Servo& servo, int angle, const char* name);

// ========== Setup ==========
void setup() {
    Serial.begin(115200);
    Serial.println("\n===== Robot Arm ESP32 Starting =====");

    // Attach servos
    servoBase.attach(PIN_BASE);
    servoUpdown.attach(PIN_UPDOWN);
    servoArm.attach(PIN_ARM);
    servoGripper.attach(PIN_GRIPPER);

    // Home position (90°)
    setServoAngle(servoBase,    90, "Base");
    setServoAngle(servoUpdown,  90, "Up/Down");
    setServoAngle(servoArm,     90, "Arm");
    setServoAngle(servoGripper, 90, "Gripper");

    // Connect WiFi
    connectWiFi();

    // Setup WebSocket
    webSocket.begin(WS_HOST, WS_PORT, WS_PATH);
    webSocket.onEvent(webSocketEvent);
    webSocket.setReconnectInterval(3000);  // Auto-reconnect every 3s
}

// ========== Loop ==========
void loop() {
    webSocket.loop();
}

// ========== WiFi ==========
void connectWiFi() {
    Serial.print("Connecting to WiFi");
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi connected");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
}

// ========== WebSocket Event Handler ==========
void webSocketEvent(WStype_t type, uint8_t* payload, size_t length) {
    switch (type) {
        case WStype_DISCONNECTED:
            wsConnected = false;
            Serial.println("[WS] Disconnected");
            break;

        case WStype_CONNECTED:
            wsConnected = true;
            Serial.println("[WS] Connected to server");
            // Identify ourselves
            webSocket.sendTXT("ESP32");
            break;

        case WStype_TEXT: {
            // Ensure null-terminated string
            char msg[64];
            size_t len = min(length, (size_t)63);
            memcpy(msg, payload, len);
            msg[len] = '\0';
            Serial.print("[WS] RX: ");
            Serial.println(msg);
            handleCommand(msg);
            break;
        }

        case WStype_BIN:
            // Binary not used
            break;

        case WStype_ERROR:
            Serial.println("[WS] Error");
            break;

        case WStype_PING:
            break;

        case WStype_PONG:
            break;
    }
}

// ========== Command Handler ==========
void handleCommand(const char* msg) {
    // Parse "servo_name:value" format
    const char* colon = strchr(msg, ':');
    if (colon == NULL) {
        Serial.println("Invalid format (no colon)");
        return;
    }

    // Extract servo name
    char servo[16];
    int nameLen = colon - msg;
    if (nameLen > 15) nameLen = 15;
    memcpy(servo, msg, nameLen);
    servo[nameLen] = '\0';

    // Extract value
    const char* valueStr = colon + 1;

    // Emergency commands
    if (strcmp(servo, "emergency") == 0) {
        if (strcmp(valueStr, "STOP") == 0) {
            emergencyStop = true;
            Serial.println(">>> EMERGENCY STOP <<<");
            servoBase.detach();
            servoUpdown.detach();
            servoArm.detach();
            servoGripper.detach();
        } else if (strcmp(valueStr, "RESUME") == 0) {
            emergencyStop = false;
            Serial.println(">>> EMERGENCY RESUME <<<");
            servoBase.attach(PIN_BASE);
            servoUpdown.attach(PIN_UPDOWN);
            servoArm.attach(PIN_ARM);
            servoGripper.attach(PIN_GRIPPER);
            setServoAngle(servoBase,    90, "Base");
            setServoAngle(servoUpdown,  90, "Up/Down");
            setServoAngle(servoArm,     90, "Arm");
            setServoAngle(servoGripper, 90, "Gripper");
        }
        return;
    }

    // Ignore commands if emergency stop is active
    if (emergencyStop) {
        Serial.println("Emergency mode - ignoring command");
        return;
    }

    // Parse angle
    int angle = atoi(valueStr);
    if (angle < 0 || angle > 180) {
        Serial.println("Invalid angle (0-180)");
        return;
    }

    // Route to correct servo
    if (strcmp(servo, "base") == 0) {
        setServoAngle(servoBase, angle, "Base");
    } else if (strcmp(servo, "updown") == 0) {
        setServoAngle(servoUpdown, angle, "Up/Down");
    } else if (strcmp(servo, "arm") == 0) {
        setServoAngle(servoArm, angle, "Arm");
    } else if (strcmp(servo, "gripper") == 0) {
        setServoAngle(servoGripper, angle, "Gripper");
    } else {
        Serial.print("Unknown servo: ");
        Serial.println(servo);
    }
}

// ========== Servo Helper ==========
void setServoAngle(Servo& servo, int angle, const char* name) {
    if (angle < 0) angle = 0;
    if (angle > 180) angle = 180;
    servo.write(angle);
    Serial.printf("Servo %s -> %d°\n", name, angle);
}
