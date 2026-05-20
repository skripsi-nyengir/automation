// ========== WebSocket ==========
const WS_PATH = '/ws';
const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
const wsUrl = `${protocol}://${window.location.host}${WS_PATH}`;

let ws = null;
let reconnectTimer = null;
let esp32Connected = false;

// ========== DOM Elements ==========
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const esp32Badge = document.getElementById('esp32Badge');

const sliders = {
    base:    document.getElementById('slider-base'),
    updown:  document.getElementById('slider-updown'),
    arm:     document.getElementById('slider-arm'),
    gripper: document.getElementById('slider-gripper'),
};

const values = {
    base:    document.getElementById('val-base'),
    updown:  document.getElementById('val-updown'),
    arm:     document.getElementById('val-arm'),
    gripper: document.getElementById('val-gripper'),
};

const btnHome = document.getElementById('btnHome');
const btnEmergency = document.getElementById('btnEmergency');
const btnResume = document.getElementById('btnResume');

// ========== Connection ==========
function connect() {
    if (ws && ws.readyState === WebSocket.OPEN) return;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('[WS] Connected');
        setConnected(true);
        if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
        }
    };

    ws.onclose = () => {
        console.log('[WS] Disconnected');
        setConnected(false);
        scheduleReconnect();
    };

    ws.onerror = () => {
        // onclose fires right after, so we just log here
        console.log('[WS] Error');
    };

    ws.onmessage = (event) => {
        const msg = event.data;

        // ESP32 status from server
        if (msg.startsWith('esp32status:')) {
            const status = msg.split(':')[1];
            esp32Connected = status === 'connected';
            updateEsp32Badge();
            return;
        }

        console.log('[WS] Message:', msg);
    };
}

function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        console.log('[WS] Reconnecting...');
        connect();
    }, 3000);
}

function send(msg) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(msg);
    }
}

// ========== Status ==========
function setConnected(connected) {
    statusDot.classList.toggle('connected', connected);
    statusText.textContent = connected ? 'Connected' : 'Disconnected';
    const disabled = !connected;
    Object.values(sliders).forEach(s => s.disabled = disabled);
    btnHome.disabled = disabled;
    btnEmergency.disabled = disabled;
    btnResume.disabled = disabled;

    if (!connected) {
        esp32Connected = false;
        updateEsp32Badge();
    }
}

function updateEsp32Badge() {
    esp32Badge.textContent = `ESP32: ${esp32Connected ? '✅ Online' : '❌ Offline'}`;
    esp32Badge.className = 'esp32-badge ' + (esp32Connected ? 'esp32-online' : 'esp32-offline');
}

// ========== Slider Controls ==========
Object.keys(sliders).forEach(key => {
    sliders[key].addEventListener('input', () => {
        const val = sliders[key].value;
        values[key].textContent = val + '°';
        send(`${key}:${val}`);
    });
});

btnHome.addEventListener('click', () => {
    const homeVal = '90';
    Object.keys(sliders).forEach(key => {
        sliders[key].value = homeVal;
        values[key].textContent = homeVal + '°';
        send(`${key}:${homeVal}`);
    });
});

btnEmergency.addEventListener('click', () => {
    send('emergency:STOP');
    btnEmergency.textContent = '✅ STOP Sent!';
    setTimeout(() => { btnEmergency.textContent = '🛑 STOP'; }, 2000);
});

btnResume.addEventListener('click', () => {
    send('emergency:RESUME');
    btnResume.textContent = '✅ Resumed!';
    setTimeout(() => { btnResume.textContent = '▶️ Resume'; }, 2000);
});

// ========== Joystick (Gamepad API) ==========
const AXIS_MAP = [
    { key: 'base',    axis: 0, invert: false },
    { key: 'updown',  axis: 1, invert: true  },
    { key: 'arm',     axis: 2, invert: false },
    { key: 'gripper', axis: 3, invert: true  },
];

const jsStatus = document.getElementById('jsStatus');
const jsName = document.getElementById('jsName');
const axisBars = {};
const axisVals = {};
for (let i = 0; i < 4; i++) {
    axisBars[i] = document.getElementById('axisBar' + i);
    axisVals[i] = document.getElementById('axisVal' + i);
}
const deadzoneSlider = document.getElementById('deadzoneSlider');
const deadzoneVal = document.getElementById('deadzoneVal');

let gamepadIndex = null;
let lastAngles = { base: 90, updown: 90, arm: 90, gripper: 90 };
let joystickActive = false;
let polling = false;

deadzoneSlider.addEventListener('input', () => {
    deadzoneVal.textContent = deadzoneSlider.value;
});

window.addEventListener('gamepadconnected', (e) => {
    gamepadIndex = e.gamepad.index;
    jsStatus.textContent = '✅ Connected';
    jsName.textContent = e.gamepad.id;
    jsStatus.style.color = '#4CAF50';
    joystickActive = true;
    if (!polling) startJoystickLoop();
});

window.addEventListener('gamepaddisconnected', () => {
    gamepadIndex = null;
    jsStatus.textContent = '❌ Not connected';
    jsName.textContent = '';
    jsStatus.style.color = '#f44336';
    joystickActive = false;
    resetAxisBars();
});

function resetAxisBars() {
    for (let i = 0; i < 4; i++) {
        axisBars[i].style.width = '50%';
        axisBars[i].style.background = '#555';
        axisVals[i].textContent = '90°';
    }
}

function axisToServo(value) {
    const v = Math.max(-1, Math.min(1, value));
    return Math.round((v + 1) * 90);
}

function applyDeadzone(value, deadzone) {
    if (Math.abs(value) < deadzone / 100) return 0;
    return value;
}

function startJoystickLoop() {
    polling = true;
    function poll() {
        if (!joystickActive || gamepadIndex === null) {
            polling = false;
            return;
        }
        const gamepads = navigator.getGamepads();
        const gp = gamepads[gamepadIndex];
        if (!gp) { polling = false; return; }

        const dz = parseFloat(deadzoneSlider.value) / 100;
        let changed = false;

        AXIS_MAP.forEach(({ key, axis, invert }) => {
            const raw = gp.axes[axis] !== undefined ? gp.axes[axis] : 0;
            const val = applyDeadzone(invert ? -raw : raw, dz);
            const angle = axisToServo(val);

            const barEl = axisBars[AXIS_MAP.findIndex(a => a.key === key)];
            const valEl = axisVals[AXIS_MAP.findIndex(a => a.key === key)];

            if (barEl) {
                barEl.style.width = angle + '%';
                barEl.style.background = angle <= 90
                    ? `hsl(${120 + (angle/90)*60}, 80%, 50%)`
                    : `hsl(${(180-angle)/90*60}, 80%, 50%)`;
            }
            if (valEl) valEl.textContent = angle + '°';

            if (angle !== lastAngles[key]) {
                lastAngles[key] = angle;
                changed = true;
                sliders[key].value = angle;
                values[key].textContent = angle + '°';
                send(`${key}:${angle}`);
            }
        });

        requestAnimationFrame(poll);
    }
    requestAnimationFrame(poll);
}

// Check for already-connected gamepad on load
window.addEventListener('load', () => {
    const gamepads = navigator.getGamepads();
    for (const gp of gamepads) {
        if (gp) {
            gamepadIndex = gp.index;
            jsStatus.textContent = '✅ Connected';
            jsName.textContent = gp.id;
            jsStatus.style.color = '#4CAF50';
            joystickActive = true;
            if (!polling) startJoystickLoop();
            break;
        }
    }
});

// ========== Start ==========
connect();
