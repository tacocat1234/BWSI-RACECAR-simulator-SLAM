"""
MIT BWSI Autonomous RACECAR
GNU General Public License v3.0
racecar-neo-outreach-labs

File Name: test_sensors.py

Title: Sensor and actuator checks

Purpose: Exercise the RACECAR Neo V2 sensor and actuator API (encoder, battery,
magnetometer, dot matrix, LED strip) and report each check as PASS or FAIL. Runs
in RacecarSim and on the physical car.

Usage:
    python test_sensors.py -s                      interactive (RacecarSim)
    python test_sensors.py                         interactive (physical car)
    python test_sensors.py -s --section encoder    run one section, print the report, exit

Interactive controls, in user program mode (controller button, RacecarSim
keyboard key in brackets):
    A    [1] api      every sensor read and actuator command once
    B    [2] encoder  drive at speed 0.25, 0.5, 1.0 (max speed 1.0) and compare;
                      needs about 25 m of straight road (RacecarSim: Long
                      Hallway Sandbox)
    X    [3] power    battery voltage and current at idle and while driving
                      (about 3 m of room ahead)
    LB   [Z] imu      gravity at rest, magnetometer strength, and compass
                      heading against the gyro through a full left circle
                      (about 1 m of room around the car)
    RB   [/] matrix   checkerboard, border, a counter, scrolling text, then
                      the text held without resending; watch the matrix
                      (RacecarSim: Settings, Show dot matrix)
    LJOY [5] led      rainbow chase, one LED walking from 0 (front-left end) to
                      83 (front-right end), solid blue, then clear; watch the
                      light band
    Y    [4] print the menu again
    Triggers and left joystick (RacecarSim: Shift keys and WASD) drive the car
    between sections.

Expected Outcome: every check in the chosen section prints PASS.
"""

import argparse
import colorsys
import math
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

import racecar_core

# add_help=False: racecar_core uses -h for headless mode
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--section", help="run one section and exit")
parser.add_argument("--angle", type=float, default=0.0, help="steering for driving sections")
args, _ = parser.parse_known_args()

rc = racecar_core.create_racecar()

FULL_SPEED = 4.0  # m/s at speed 1.0 and max speed 1.0


class Report:
    """PASS/FAIL rows for one section."""

    def __init__(self, title: str) -> None:
        self.title = title
        self.rows: List[Tuple[str, bool, str]] = []

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        self.rows.append((name, bool(passed), detail))

    def print(self) -> None:
        width = max([len(name) for name, _, _ in self.rows] + [5])
        print(f"\n== {self.title}")
        for name, passed, detail in self.rows:
            print(f"  {'PASS' if passed else 'FAIL'}  {name.ljust(width)}  {detail}")
        failed = sum(1 for _, passed, _ in self.rows if not passed)
        print(f"  {len(self.rows) - failed}/{len(self.rows)} passed")


class Section:
    """A check that runs over one or more frames. step() returns True when finished."""

    title = ""

    def __init__(self) -> None:
        self.report = Report(self.title)
        self.start_time = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    def step(self) -> bool:
        raise NotImplementedError


def finite(values: Any) -> bool:
    return bool(np.all(np.isfinite(np.asarray(values, dtype=float))))


class ApiSection(Section):
    """Calls every sensor read and actuator command once and checks the result types."""

    title = "api: sensor reads and actuator commands"

    def step(self) -> bool:
        r = self.report
        for name, read in (("linear acceleration", rc.physics.get_linear_acceleration),
                           ("angular velocity", rc.physics.get_angular_velocity),
                           ("magnetic field", rc.physics.get_magnetic_field)):
            value = read()
            r.check(f"{name} is 3 floats", value is not None and np.shape(value) == (3,) and finite(value),
                    np.array2string(np.asarray(value), precision=6) if value is not None else "None")
        for name, read in (("encoder speed", rc.physics.get_encoder_speed),
                           ("battery voltage", rc.physics.get_battery_voltage),
                           ("battery current", rc.physics.get_battery_current)):
            value = read()
            r.check(f"{name} is a float", isinstance(value, float) and finite(value), f"{value:.4f}")
        r.check("battery current not negative", rc.physics.get_battery_current() >= 0)

        samples = rc.lidar.get_samples()
        r.check("lidar samples", len(samples) == rc.lidar.get_num_samples() == 1080, f"{len(samples)}")

        matrix = rc.display.new_matrix()
        r.check("new_matrix shape", matrix.shape == (8, 24), f"{matrix.shape}")
        matrix[::2, ::2] = 1
        matrix[1::2, 1::2] = 1
        rc.display.set_matrix(matrix)
        r.check("set_matrix / get_matrix round trip", np.array_equal(rc.display.get_matrix(), matrix))
        rc.display.show_text("API OK")

        r.check("LED count", rc.led.get_num_pixels() == 84, f"{rc.led.get_num_pixels()}")
        colors = [(i * 3 % 256, 255 - i * 3 % 256, 40) for i in range(rc.led.get_num_pixels())]
        rc.led.set_pixels(colors)
        r.check("set_pixels / get_pixels round trip", [tuple(c) for c in rc.led.get_pixels()] == colors)
        rc.led.set_pixel(0, (255, 0, 0))
        r.check("set_pixel", tuple(rc.led.get_pixels()[0]) == (255, 0, 0))
        return True


class EncoderSection(Section):
    """Holds three speed commands at max speed 1.0 and compares the encoder with the target."""

    title = "encoder: steady speed against the command (max speed 1.0)"
    # (speed command, seconds); the last second of each step is averaged
    STEPS = [(0.25, 3.0), (0.5, 3.0), (1.0, 3.0), (0.0, 3.0)]
    # The speed loop overshoots a step by about 6% and settles over several
    # seconds, and the encoder reads wheel speed, which includes about 1% slip
    TOLERANCE = 0.05

    def __init__(self) -> None:
        super().__init__()
        self.index = 0
        self.step_start = time.monotonic()
        self.samples: List[float] = []
        rc.drive.set_max_speed(1.0)

    def step(self) -> bool:
        command, seconds = self.STEPS[self.index]
        rc.drive.set_speed_angle(command, args.angle if command else 0.0)
        t = time.monotonic() - self.step_start
        if t > seconds - 1.0:
            self.samples.append(rc.physics.get_encoder_speed())
        if t < seconds:
            return False

        target = command * FULL_SPEED
        mean = float(np.mean(self.samples)) if self.samples else float("nan")
        spread = float(np.std(self.samples)) if self.samples else float("nan")
        if command == 0:
            self.report.check("stopped", abs(mean) < 0.05, f"mean {mean:.3f} m/s")
        else:
            error = (mean - target) / target
            self.report.check(f"speed {command:.2f} -> {target:.1f} m/s", abs(error) <= self.TOLERANCE,
                              f"mean {mean:.3f} m/s, std {spread:.3f}, error {error:+.1%}")
        self.index += 1
        self.samples = []
        self.step_start = time.monotonic()
        if self.index < len(self.STEPS):
            return False
        rc.drive.set_speed_angle(0, 0)
        return True


class PowerSection(Section):
    """Compares battery voltage and current at idle and while driving with the sim's model."""

    title = "power: battery voltage and current"
    IDLE_S = 2.0
    DRIVE_S = 2.5
    SPEED = 0.25  # 1.0 m/s at max speed 1.0

    def __init__(self) -> None:
        super().__init__()
        self.idle: List[Tuple[float, float]] = []
        self.drive: List[Tuple[float, float, float]] = []
        rc.drive.set_max_speed(1.0)

    def step(self) -> bool:
        t = self.elapsed()
        volts = rc.physics.get_battery_voltage()
        amps = rc.physics.get_battery_current()
        if t < self.IDLE_S:
            rc.drive.set_speed_angle(0, 0)
            if t > 0.5:
                self.idle.append((volts, amps))
            return False
        if t < self.IDLE_S + self.DRIVE_S:
            rc.drive.set_speed_angle(self.SPEED, args.angle)
            if t > self.IDLE_S + 1.0:
                self.drive.append((volts, amps, rc.physics.get_encoder_speed()))
            return False
        rc.drive.set_speed_angle(0, 0)

        r = self.report
        idle_v = float(np.mean([v for v, _ in self.idle]))
        idle_a = float(np.mean([a for _, a in self.idle]))
        drive_v = float(np.mean([v for v, _, _ in self.drive]))
        drive_a = float(np.mean([a for _, a, _ in self.drive]))
        speed = float(np.mean([abs(e) for _, _, e in self.drive]))
        expected_a = 5.0 + 5.0 * min(speed / FULL_SPEED, 1.0)
        readings = self.idle + [d[:2] for d in self.drive]
        r.check("voltage in 7.0 to 8.4 V", all(6.99 <= v <= 8.41 for v, _ in readings),
                f"idle {idle_v:.3f} V, driving {drive_v:.3f} V")
        r.check("current never negative", all(a >= 0 for _, a in readings))
        r.check("idle current about 2.5 A", abs(idle_a - 2.5) < 0.1, f"{idle_a:.3f} A")
        r.check("driving current 5 A + 5 A x speed / 4 m/s", abs(drive_a - expected_a) < 0.3,
                f"{drive_a:.3f} A at {speed:.2f} m/s (model {expected_a:.2f} A)")
        sag = idle_v - drive_v
        r.check("voltage sags 0.02 ohm x extra current", abs(sag - 0.02 * (drive_a - idle_a)) < 0.03,
                f"{sag * 1000:.0f} mV for {drive_a - idle_a:.2f} A")
        return True


class ImuSection(Section):
    """Gravity and field strength at rest, then the compass heading against the gyro in a left circle."""

    title = "imu: REP-103 axes, magnetometer, compass"
    REST_S = 2.0
    TURN_S = 6.0

    def __init__(self) -> None:
        super().__init__()
        self.rest_accel: List[Any] = []
        self.rest_mag: List[Any] = []
        self.gyro_heading = 0.0
        self.mag_heading: List[float] = []
        self.yaw_rates: List[float] = []
        rc.drive.set_max_speed(1.0)

    def step(self) -> bool:
        t = self.elapsed()
        mag = rc.physics.get_magnetic_field()
        if t < self.REST_S:
            rc.drive.set_speed_angle(0, 0)
            if t > 0.5:
                self.rest_accel.append(rc.physics.get_linear_acceleration())
                self.rest_mag.append(mag)
            return False
        if t < self.REST_S + self.TURN_S:
            # Full left lock at 1 m/s: one circle about every 3 s
            rc.drive.set_speed_angle(0.25, -1.0)
            if t > self.REST_S + 1.0:
                yaw_rate = rc.physics.get_angular_velocity()[2]
                self.yaw_rates.append(yaw_rate)
                self.gyro_heading += math.degrees(yaw_rate) * rc.get_delta_time()
                self.mag_heading.append(math.degrees(math.atan2(-mag[1], mag[0])))
            return False
        rc.drive.set_speed_angle(0, 0)

        r = self.report
        accel = np.mean(self.rest_accel, axis=0)
        r.check("at rest, gravity on +z", abs(accel[2] - 9.81) < 0.3 and abs(accel[0]) < 0.3 and abs(accel[1]) < 0.3,
                np.array2string(accel, precision=3))
        field_ut = np.mean(self.rest_mag, axis=0) * 1e6
        strength = float(np.linalg.norm(field_ut))
        r.check("field strength 51 uT, pointing down", abs(strength - 51.1) < 3.0 and field_ut[2] < -40,
                f"{np.array2string(field_ut, precision=2)} uT, |B| {strength:.2f}")
        r.check("left turn is positive yaw (z)", float(np.mean(self.yaw_rates)) > 0.5,
                f"{np.mean(self.yaw_rates):.3f} rad/s")
        unwrapped = np.degrees(np.unwrap(np.radians(self.mag_heading)))
        compass_turn = float(unwrapped[-1] - unwrapped[0])
        gyro_turn = self.gyro_heading
        r.check("compass heading increases turning left", compass_turn > 300,
                f"compass {compass_turn:.1f} deg")
        r.check("compass matches integrated gyro within 5 percent", abs(compass_turn - gyro_turn) < 0.05 * abs(gyro_turn),
                f"compass {compass_turn:.1f} deg, gyro {gyro_turn:.1f} deg")
        return True


class MatrixSection(Section):
    """Shows patterns, a counter, and scrolling text, then holds the text without resending."""

    title = "matrix: dot matrix frames and text"
    # (start time, what to show, what the matrix should look like)
    STEPS = [
        (0.0, "checkerboard", "alternating dots"),
        (3.0, "border", "a rectangle around the edge"),
        (6.0, "counter", "0 s to 5 s, one per second, not scrolling"),
        (12.0, "scroll", "a long message scrolling right to left, once every 4 s"),
        (20.0, "hold", "the same message still scrolling: nothing is resent"),
    ]
    END = 30.0

    def __init__(self) -> None:
        super().__init__()
        self.index = -1
        self.count = -1

    def step(self) -> bool:
        t = self.elapsed()
        r = self.report
        while self.index + 1 < len(self.STEPS) and t >= self.STEPS[self.index + 1][0]:
            self.index += 1
            name, expect = self.STEPS[self.index][1:]
            print(f"  [{t:5.1f} s] {name}: expect {expect}")
            if name == "checkerboard":
                matrix = rc.display.new_matrix()
                matrix[::2, ::2] = 1
                matrix[1::2, 1::2] = 1
                rc.display.set_matrix(matrix)
                r.check("checkerboard round trip", np.array_equal(rc.display.get_matrix(), matrix))
            elif name == "border":
                matrix = rc.display.new_matrix()
                matrix[0, :] = matrix[-1, :] = 1
                matrix[:, 0] = matrix[:, -1] = 1
                rc.display.set_matrix(matrix * 255)
                r.check("0/255 values accepted", int(rc.display.get_matrix().max()) == 255)
            elif name == "scroll":
                rc.display.show_text("Hello from the sensor checks!")
        if self.index >= 0 and self.STEPS[self.index][1] == "counter":
            count = min(int(t - 6.0), 5)
            if count != self.count:
                self.count = count
                rc.display.show_text(f"{count} s")
        if t < self.END:
            return False
        rc.display.set_matrix_intensity(0.5)
        r.check("set_matrix_intensity accepted", True, "no-op with one warning, as on the car")
        r.check("frames and text sent", True, "confirm what the matrix showed against the expectations above")
        return True


class LedSection(Section):
    """Checks the LED API, then shows a rainbow chase, a walking LED, solid blue, and clear."""

    title = "led: LED strip colors and order"
    STEPS = [
        (0.0, "chase", "a rainbow moving along the band"),
        (4.0, "walk", "one white LED from the front-left end, around the rear, to the front-right end"),
        (10.5, "blue", "the whole band blue"),
        (12.5, "clear", "the whole band dark"),
    ]
    END = 14.5

    def __init__(self) -> None:
        super().__init__()
        self.index = -1
        r = self.report
        n = rc.led.get_num_pixels()
        r.check("84 pixels", n == 84, f"{n}")
        colors = [(i, 2 * i, 255 - i) for i in range(n)]
        rc.led.set_pixels(colors)
        r.check("set_pixels / get_pixels round trip", [tuple(c) for c in rc.led.get_pixels()] == colors)
        for bad in (n, -1):
            try:
                rc.led.set_pixel(bad, (255, 255, 255))
                r.check(f"set_pixel({bad}) raises IndexError", False)
            except IndexError:
                r.check(f"set_pixel({bad}) raises IndexError", True)

    def step(self) -> bool:
        t = self.elapsed()
        while self.index + 1 < len(self.STEPS) and t >= self.STEPS[self.index + 1][0]:
            self.index += 1
            name, expect = self.STEPS[self.index][1:]
            print(f"  [{t:5.1f} s] {name}: expect {expect}")
            if name == "blue":
                rc.led.set_color((0, 0, 255))
            elif name == "clear":
                rc.led.clear()
        name = self.STEPS[self.index][1]
        n = rc.led.get_num_pixels()
        if name == "chase":
            offset = int(t * 20)
            rc.led.set_pixels([rainbow((i + offset) % n, n) for i in range(n)])
        elif name == "walk":
            lit = min(n - 1, int((t - 4.0) / 6.0 * n))
            rc.led.set_color((0, 0, 0))
            rc.led.set_pixel(lit, (255, 255, 255))
        if t < self.END:
            return False
        self.report.check("colors sent", True, "confirm the band showed each step above")
        return True


def rainbow(i: int, n: int) -> Tuple[int, int, int]:
    """A hue around the color wheel for LED i of n."""
    r, g, b = colorsys.hsv_to_rgb(i / n, 1, 1)
    return (int(r * 255), int(g * 255), int(b * 255))


SECTIONS: Dict[str, Tuple[Any, Type[Section]]] = {
    "api": (rc.controller.Button.A, ApiSection),
    "encoder": (rc.controller.Button.B, EncoderSection),
    "power": (rc.controller.Button.X, PowerSection),
    "imu": (rc.controller.Button.LB, ImuSection),
    "matrix": (rc.controller.Button.RB, MatrixSection),
    "led": (rc.controller.Button.LJOY, LedSection),
}

active: Optional[Section] = None


# RacecarSim keyboard keys for the controller buttons (Controller.cs keyboardButtonMap)
SIM_KEYS = {"A": "1", "B": "2", "X": "3", "Y": "4", "LB": "Z", "RB": "/", "LJOY": "5", "RJOY": "6"}


def print_menu() -> None:
    print("\nSensor and actuator checks. Press (RacecarSim keyboard key in brackets):")
    for name, (button, factory) in SECTIONS.items():
        print(f"  {button.name:<4} [{SIM_KEYS[button.name]}] {name:<8} {factory.title}")
    print("  Y    [4] menu")


def finish() -> None:
    """Leave through the library's SIGINT path, which tells RacecarSim the program exited."""
    os.kill(os.getpid(), signal.SIGINT)


def start() -> None:
    global active
    rc.drive.stop()
    if args.section:
        active = SECTIONS[args.section][1]()
    else:
        print_menu()


def update() -> None:
    global active
    if active is not None:
        if active.step():
            active.report.print()
            active = None
            if args.section:
                finish()
        return

    for name, (button, factory) in SECTIONS.items():
        if rc.controller.was_pressed(button):
            print(f"\nRunning {name}...")
            active = factory()
            return
    if rc.controller.was_pressed(rc.controller.Button.Y):
        print_menu()

    # Manual driving between sections
    speed = rc.controller.get_trigger(rc.controller.Trigger.RIGHT) - rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    angle = rc.controller.get_joystick(rc.controller.Joystick.LEFT)[0]
    rc.drive.set_speed_angle(speed, angle)


if __name__ == "__main__":
    if args.section and args.section not in SECTIONS:
        print(f"Unknown section {args.section}; choose from {', '.join(SECTIONS)}")
        sys.exit(2)
    rc.set_start_update(start, update)
    rc.go()
