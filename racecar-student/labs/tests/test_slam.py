"""
MIT BWSI Autonomous RACECAR
GNU General Public License v3.0
racecar-neo-outreach-labs

File Name: test_slam.py

Title: SLAM checks

Purpose: Exercise rc.slam (mode, map, pose, localized, save) against the car's
autonomy service and report each check as PASS or FAIL. Physical car only;
RacecarSim has no SLAM.

Before running: start mapping, or load a saved map, in the map viewer
(http://<car>:8084) or with `racecar map start NAME` / `racecar map load NAME`.

Usage:
    python test_slam.py                       interactive
    python test_slam.py --section status      run one section, print the report, exit

Interactive controls, in user program mode:
    A    status  mode, map name, localized, and the pose inside the map
    B    drive   drive about 3 m with the triggers and joystick within 20 s;
                 the SLAM pose must follow the encoder distance and stay
                 localized
    X    save    save the map (mapping only)
    LB   lift    lift the car and carry it about 2 m without touching the
                 controls; localized must turn false within 10 s
    Y    print the menu again
    Triggers and left joystick drive the car between sections.

Expected Outcome: every check in the chosen section prints PASS.
"""

import argparse
import math
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type

import racecar_core

# add_help=False: racecar_core uses -h for headless mode
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--section", help="run one section and exit")
args, _ = parser.parse_known_args()

rc = racecar_core.create_racecar()


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


def manual_drive() -> None:
    speed = rc.controller.get_trigger(rc.controller.Trigger.RIGHT) - rc.controller.get_trigger(
        rc.controller.Trigger.LEFT
    )
    angle = rc.controller.get_joystick(rc.controller.Joystick.LEFT)[0]
    rc.drive.set_speed_angle(speed, angle)


class StatusSection(Section):
    """Mode, map name, localized, and the pose against the map."""

    title = "status: mode, map, localized, pose"

    def step(self) -> bool:
        r = self.report
        mode = rc.slam.get_mode()
        r.check("autonomy running", mode in ("mapping", "localization"), f"mode {mode!r}")
        if mode == "off":
            print("  Start mapping or load a map in the map viewer (port 8084), then run again.")
            return True
        r.check("map name", rc.slam.get_map_name() != "", repr(rc.slam.get_map_name()))
        r.check("localized", rc.slam.is_localized(), "")
        pose = rc.slam.get_pose()
        r.check("pose", pose is not None, "" if pose is None else
                f"x {pose[0]:.2f} m, y {pose[1]:.2f} m, heading {pose[2]:.0f} deg")
        m = rc.slam.get_map()
        r.check("map", m is not None, repr(m))
        if pose is not None and m is not None:
            r.check("pose inside the map", m.to_cell(pose[0], pose[1]) is not None, "")
            r.check("car stands on free space", m.value_at(pose[0], pose[1]) != m.WALL,
                    f"value {m.value_at(pose[0], pose[1])}")
            r.check("heading in [-180, 180)", -180.0 <= pose[2] < 180.0, f"{pose[2]:.1f} deg")
        return True


class DriveSection(Section):
    """Compares the distance the SLAM pose moved with the encoder's while the student drives."""

    title = "drive: the pose follows the car"
    DURATION_S = 20.0
    MIN_DISTANCE_M = 1.5
    # SLAM corrects the odometry, so the two distances agree only roughly
    TOLERANCE = 0.25

    def __init__(self) -> None:
        super().__init__()
        self.last_pose: Optional[Tuple[float, float, float]] = None
        self.slam_distance = 0.0
        self.encoder_distance = 0.0
        self.frames = 0
        self.localized_frames = 0
        print(f"  Drive about 3 m with the triggers and joystick within {self.DURATION_S:.0f} s.")

    def step(self) -> bool:
        manual_drive()
        self.encoder_distance += abs(rc.physics.get_encoder_speed()) * rc.get_delta_time()
        pose = rc.slam.get_pose()
        self.frames += 1
        self.localized_frames += rc.slam.is_localized()
        if pose is not None and self.last_pose is not None:
            self.slam_distance += math.hypot(pose[0] - self.last_pose[0], pose[1] - self.last_pose[1])
        self.last_pose = pose
        if self.elapsed() < self.DURATION_S:
            return False
        rc.drive.stop()
        r = self.report
        r.check("drove far enough", self.encoder_distance >= self.MIN_DISTANCE_M,
                f"encoder {self.encoder_distance:.2f} m")
        error = (self.slam_distance - self.encoder_distance) / max(self.encoder_distance, 1e-6)
        r.check("pose moved with the car", abs(error) <= self.TOLERANCE,
                f"pose {self.slam_distance:.2f} m, encoder {self.encoder_distance:.2f} m, {error:+.0%}")
        share = self.localized_frames / max(self.frames, 1)
        r.check("stayed localized", share >= 0.95, f"{share:.0%} of frames")
        return True


class SaveSection(Section):
    """Saves the map being built and waits for the result."""

    title = "save: save_map while mapping"
    TIMEOUT_S = 30.0

    def __init__(self) -> None:
        super().__init__()
        self.started = False

    def step(self) -> bool:
        r = self.report
        if not self.started:
            if rc.slam.get_mode() != "mapping":
                rc.slam.save_map()
                r.check("refused unless mapping", rc.slam.get_save_status() == "failed: not mapping",
                        rc.slam.get_save_status())
                print("  Start mapping to test a save.")
                return True
            rc.slam.save_map()
            r.check("returns at once", rc.slam.get_save_status() == "saving", rc.slam.get_save_status())
            self.started = True
            return False
        status = rc.slam.get_save_status()
        if status == "saving" and self.elapsed() < self.TIMEOUT_S:
            return False
        r.check("saved", status == "saved", f"{status!r} after {self.elapsed():.1f} s")
        return True


class LiftSection(Section):
    """Checks that a car carried to another spot stops reporting localized."""

    title = "lift: localized turns false when the car is carried"
    TIMEOUT_S = 10.0

    def __init__(self) -> None:
        super().__init__()
        self.was_localized = rc.slam.is_localized()
        rc.drive.stop()
        print("  Lift the car now and carry it about 2 m. Do not touch the controls.")

    def step(self) -> bool:
        rc.drive.stop()
        r = self.report
        if not self.was_localized:
            r.check("localized before the lift", False, "localize first, then run again")
            return True
        if rc.slam.is_localized() and self.elapsed() < self.TIMEOUT_S:
            return False
        lost = not rc.slam.is_localized()
        r.check("localized turned false", lost, f"after {self.elapsed():.1f} s")
        if lost:
            print("  Set the car down, then use Set pose or Find me in the map viewer.")
        return True


SECTIONS: Dict[str, Tuple[Any, Type[Section]]] = {
    "status": (rc.controller.Button.A, StatusSection),
    "drive": (rc.controller.Button.B, DriveSection),
    "save": (rc.controller.Button.X, SaveSection),
    "lift": (rc.controller.Button.LB, LiftSection),
}

active: Optional[Section] = None


def print_menu() -> None:
    print("\nSLAM checks. Press:")
    for name, (button, factory) in SECTIONS.items():
        print(f"  {button.name:<4} {name:<7} {factory.title}")
    print("  Y    menu")


def finish() -> None:
    """Leave through the library's SIGINT path."""
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
    manual_drive()


if __name__ == "__main__":
    if args.section and args.section not in SECTIONS:
        print(f"Unknown section {args.section}; choose from {', '.join(SECTIONS)}")
        sys.exit(2)
    rc.set_start_update(start, update)
    rc.go()
