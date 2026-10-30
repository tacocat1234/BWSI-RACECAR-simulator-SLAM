"""
MIT BWSI Autonomous RACECAR
GNU General Public License v3.0
racecar-neo-outreach-labs

File Name: dashboard.py

Title: Sensor dashboard

Purpose: A browser dashboard for every racecar_core sensor and control, for
testing in RacecarSim or on the physical car. Adapted from the RACECAR Neo web
teleop dashboard (MITRacecarNeo/teleop_dashboard, branch racecar-neo), with
the ROS side replaced by racecar_core calls, so the same program runs against
the simulator and the car.

Usage:
    python dashboard.py -s          RacecarSim (enter user program mode to start)
    python dashboard.py             physical car
    then open http://localhost:8081 in a browser.

Options:
    --port N        HTTP port (default 8081)
    --camera-hz F   color and depth image rate (default 10)

Threading: update() is the only code that calls racecar_core. The HTTP server
thread reads the shared state and queues commands; update() applies them. A
drive command older than 0.5 s reads as released, so a closed tab or a dropped
link coasts the car to a stop.
"""

import argparse
import json
import math
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, List, Tuple

import cv2
import numpy as np

import racecar_core

# add_help=False: racecar_core uses -h for headless mode
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--port", type=int, default=8081)
parser.add_argument("--camera-hz", type=float, default=10.0)
args, _ = parser.parse_known_args()

rc = racecar_core.create_racecar()

BASE = Path(__file__).resolve().parent
CMD_TIMEOUT_S = 0.5
HISTORY_S = 10.0
PREVIEW_WIDTH = 368
JPEG_QUALITY = 70
DEPTH_FAR_M = 4.0
DEPTH_NEAR_M = 0.15
DEPTH_FLOOR = 24
FULL_SPEED_M_S = 4.0  # at speed 1.0 and max speed 1.0

_lock = threading.Lock()
_state: Dict[str, Any] = {"running": False}
_color_jpeg = b""
_depth_jpeg = b""
_cmd: Dict[str, Any] = {
    "fwd": 0, "turn": 0, "stamp": 0.0,
    "speed": 0.5, "angle": 1.0, "max_speed": 0.25, "max_speed_changed": True,
}
_hist: Deque[List[float]] = deque()
# Matrix and LED commands from the browser, applied by update()
_actions: Deque[Tuple[str, Any]] = deque(maxlen=20)
_T0 = time.monotonic()
_last_camera = 0.0
_frame_times: Deque[float] = deque(maxlen=60)


def _now() -> float:
    return time.monotonic() - _T0


def _axis(value: Any) -> int:
    v = int(value)
    return (v > 0) - (v < 0)


def _colorize_depth(meters: np.ndarray) -> np.ndarray:
    """Bright is near, dark is far, black is no return (inferno ramp)."""
    valid = meters > 0
    norm = np.clip((meters - DEPTH_NEAR_M) / (DEPTH_FAR_M - DEPTH_NEAR_M), 0.0, 1.0)
    ramp = (DEPTH_FLOOR + (1.0 - norm) * (255 - DEPTH_FLOOR)).astype(np.uint8)
    out = cv2.applyColorMap(ramp, cv2.COLORMAP_INFERNO)
    out[~valid] = (0, 0, 0)
    return out


def _jpeg(image: np.ndarray, nearest: bool = False) -> bytes:
    h, w = image.shape[:2]
    size = (PREVIEW_WIDTH, max(1, round(PREVIEW_WIDTH * h / w)))
    small = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)
    return cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()


def _read_camera() -> None:
    """Color and depth previews at --camera-hz."""
    global _color_jpeg, _depth_jpeg, _last_camera
    now = time.monotonic()
    if now - _last_camera < 1.0 / args.camera_hz:
        return
    _last_camera = now
    color = rc.camera.get_color_image()
    depth = rc.camera.get_depth_image()
    color_jpeg = _jpeg(color) if color is not None else b""
    # racecar_core depth is in cm
    depth_jpeg = _jpeg(_colorize_depth(np.asarray(depth, dtype=np.float32) / 100.0), True) if depth is not None else b""
    with _lock:
        _color_jpeg = color_jpeg
        _depth_jpeg = depth_jpeg


def _apply_drive() -> Dict[str, float]:
    with _lock:
        cmd = dict(_cmd)
        _cmd["max_speed_changed"] = False
    held = cmd["fwd"] != 0 or cmd["turn"] != 0
    timed_out = held and time.monotonic() - cmd["stamp"] > CMD_TIMEOUT_S
    fwd, turn = (0, 0) if timed_out else (cmd["fwd"], cmd["turn"])
    if cmd["max_speed_changed"]:
        rc.drive.set_max_speed(cmd["max_speed"])
    speed = max(-1.0, min(1.0, fwd * cmd["speed"]))
    angle = max(-1.0, min(1.0, turn * cmd["angle"]))
    rc.drive.set_speed_angle(speed, angle)
    return {"speed_cmd": speed, "angle_cmd": angle, "held": float(held and not timed_out),
            "timed_out": float(timed_out), "target": speed * cmd["max_speed"] * FULL_SPEED_M_S}


def _apply_actions() -> None:
    while True:
        with _lock:
            if not _actions:
                return
            kind, value = _actions.popleft()
        if kind == "matrix":
            rc.display.set_matrix(np.array(value, dtype=np.uint8).reshape((8, 24)))
        elif kind == "text":
            rc.display.show_text(value)
        elif kind == "leds":
            rc.led.set_pixels([tuple(c) for c in value])


def start() -> None:
    with _lock:
        _cmd["max_speed_changed"] = True
        _state["running"] = True
    print(f">> Dashboard on http://localhost:{args.port}")


def update() -> None:
    now = _now()
    _frame_times.append(time.monotonic())
    drive = _apply_drive()
    _apply_actions()
    _read_camera()

    accel = rc.physics.get_linear_acceleration()
    gyro = rc.physics.get_angular_velocity()
    encoder = rc.physics.get_encoder_speed()
    volts = rc.physics.get_battery_voltage()
    amps = rc.physics.get_battery_current()
    mag_ut = [float(v) * 1e6 for v in rc.physics.get_magnetic_field()]
    # 0 facing magnetic north, increasing counterclockwise
    heading = math.degrees(math.atan2(-mag_ut[1], mag_ut[0]))
    samples = np.asarray(rc.lidar.get_samples(), dtype=np.float32)
    n = len(samples)
    # Student angle: 0 the nose, positive to the right; racecar_core lidar is in cm
    scan = [[round(i * 360.0 / n, 1), round(float(samples[i]) / 100.0, 3)]
            for i in range(0, n, 3) if samples[i] > 0]

    _hist.append([round(now, 2), round(encoder, 3), round(drive["target"], 3),
                  *[round(float(v), 3) for v in accel], *[round(float(v), 4) for v in gyro],
                  round(volts, 3), round(amps, 3), *[round(v, 2) for v in mag_ut]])
    while _hist and now - _hist[0][0] > HISTORY_S:
        _hist.popleft()
    fps = (len(_frame_times) - 1) / (_frame_times[-1] - _frame_times[0]) if len(_frame_times) > 1 else 0.0

    with _lock:
        _state.update({
            "t": round(now, 2), "fps": round(fps, 1),
            **{k: round(v, 3) for k, v in drive.items()},
            "max_speed": _cmd["max_speed"], "speed": _cmd["speed"], "angle": _cmd["angle"],
            "accel": [round(float(v), 3) for v in accel],
            "gyro": [round(float(v), 4) for v in gyro],
            "encoder": round(encoder, 3),
            "volts": round(volts, 3), "amps": round(amps, 3),
            "mag": [round(v, 2) for v in mag_ut], "heading": round(heading, 1),
            "scan": scan,
            "matrix": [int(v != 0) for v in np.asarray(rc.display.get_matrix()).flatten()],
            "leds": [list(c) for c in rc.led.get_pixels()],
            "hist": list(_hist),
        })


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path == "/":
            self._send((BASE / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path.startswith("/color") or self.path.startswith("/depth"):
            with _lock:
                data = _color_jpeg if self.path.startswith("/color") else _depth_jpeg
            if data:
                self._send(data, "image/jpeg", cache="no-store")
            else:
                self.send_error(503)
        elif self.path == "/state":
            with _lock:
                body = json.dumps(_state)
            self._send(body.encode(), "application/json", cache="no-store")
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        data = self._body()
        with _lock:
            if self.path == "/cmd":
                _cmd["fwd"] = _axis(data.get("fwd", 0))
                _cmd["turn"] = _axis(data.get("turn", 0))
                _cmd["stamp"] = time.monotonic()
            elif self.path == "/params":
                for key in ("speed", "angle", "max_speed"):
                    if key in data:
                        _cmd[key] = max(0.0, min(1.0, float(data[key])))
                _cmd["max_speed_changed"] = "max_speed" in data or _cmd["max_speed_changed"]
            elif self.path == "/matrix":
                pixels = [1 if v else 0 for v in data.get("pixels", [])][:192]
                _actions.append(("matrix", pixels + [0] * (192 - len(pixels))))
            elif self.path == "/text":
                _actions.append(("text", str(data.get("text", ""))[:255]))
            elif self.path == "/leds":
                colors = [[max(0, min(255, int(v))) for v in c[:3]] for c in data.get("colors", [])][:84]
                _actions.append(("leds", colors + [[0, 0, 0]] * (84 - len(colors))))
            elif self.path == "/stop":
                _cmd["fwd"] = _cmd["turn"] = 0
            else:
                self.send_error(404)
                return
        self._send(b"ok", "text/plain")

    def _body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length)) if length else {}

    def _send(self, body: bytes, ctype: str, cache: str = "") -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *log_args: Any) -> None:
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f">> Dashboard server on http://localhost:{args.port}; enter user program mode to start.")
    rc.set_start_update(start, update)
    rc.go()
    server.shutdown()
