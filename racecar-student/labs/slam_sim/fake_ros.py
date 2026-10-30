#lightweight ROS 2 mock so SLAM nodes can run in sim w/out needing a full ROS install.

#must call fake_ros.install() before importing any nodes; should fake ROS module and all its functionalti

#has:
#topics, with pub and sub that pushes messages like ros
#timers: soem spin (cuz main uses update loop thing)
#clock: hijack rc.getdeltatime
import sys
import types
from dataclasses import dataclass, field


#time hlerper class
class Time:
    def __init__(self, nanoseconds=0):
        self.nanoseconds = int(nanoseconds)

    @property
    def sec(self):
        return self.nanoseconds // 1_000_000_000#conversion

    @property
    def nanosec(self):
        return self.nanoseconds % 1_000_000_000

    def to_msg(self):
        return self

    def __repr__(self):
        return f"Time({self.nanoseconds} ns)"


class SimClock:
    def __init__(self):
        self.ns = 0

    def advance(self, dt_seconds):
        self.ns += int(round(dt_seconds * 1e9))

    def now(self):
        return Time(self.ns)


CLOCK = SimClock()#get sim clcok 


#message types but fake
@dataclass
class Header:
    stamp: Time = field(default_factory=Time)
    frame_id: str = ""


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


@dataclass
class Float32:
    data: float = 0.0


@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


@dataclass
class Pose:
    position: Point = field(default_factory=Point)
    orientation: Quaternion = field(default_factory=Quaternion)


@dataclass
class MapMetaData:
    map_load_time: Time = field(default_factory=Time)
    resolution: float = 0.0
    width: int = 0
    height: int = 0
    origin: Pose = field(default_factory=Pose)


@dataclass
class OccupancyGrid:
    header: Header = field(default_factory=Header)
    info: MapMetaData = field(default_factory=MapMetaData)
    data: list = field(default_factory=list)


@dataclass
class LaserScan:
    header: Header = field(default_factory=Header)
    angle_min: float = 0.0
    angle_max: float = 0.0
    angle_increment: float = 0.0
    time_increment: float = 0.0
    scan_time: float = 0.0
    range_min: float = 0.0
    range_max: float = 0.0
    ranges: list = field(default_factory=list)
    intensities: list = field(default_factory=list)


@dataclass
class AckermannDrive:
    steering_angle: float = 0.0
    steering_angle_velocity: float = 0.0
    speed: float = 0.0
    acceleration: float = 0.0
    jerk: float = 0.0


@dataclass
class AckermannDriveStamped:
    header: Header = field(default_factory=Header)
    drive: AckermannDrive = field(default_factory=AckermannDrive)


#stuff
_SUBSCRIBERS = {} #topic -> [callbacks]
_TIMERS = [] #kinda scuffed globals but wtver


class _Publisher:
    def __init__(self, topic):
        self.topic = topic

    def publish(self, msg):
        for cb in list(_SUBSCRIBERS.get(self.topic, [])):#for
            cb(msg) #send message to every subscriber of this topic


class _Timer: #tiemr
    def __init__(self, period_s, callback):
        self.period_ns = int(period_s * 1e9)
        self.callback = callback
        self.next_ns = CLOCK.ns + self.period_ns


class _Logger: #fake logger for debugs
    def __init__(self, name):
        self.name = name
        self._last = {}

    def _emit(self, level, msg):
        last = self._last.get(msg)
        if last is None or CLOCK.ns - last >= 1_000_000_000:
            self._last[msg] = CLOCK.ns
            print(f"[{level}] [{CLOCK.ns / 1e9:8.2f}] [{self.name}]: {msg}")

    def debug(self, msg): pass
    def info(self, msg): self._emit("INFO", msg)
    def warn(self, msg): self._emit("WARN", msg)
    warning = warn
    def error(self, msg): self._emit("ERROR", msg)


class Node: #fake ros node
    def __init__(self, node_name):
        self._name = node_name
        self._logger = _Logger(node_name)

    def create_subscription(self, msg_type, topic, callback, qos_profile=10):
        _SUBSCRIBERS.setdefault(topic, []).append(callback) #register callback for topic
        return callback

    def create_publisher(self, msg_type, topic, qos_profile=10):
        return _Publisher(topic) #create publisher for topic

    def create_timer(self, period_sec, callback):
        timer = _Timer(period_sec, callback) #create timer with period and callback
        _TIMERS.append(timer)
        return timer

    def get_clock(self):
        return CLOCK

    def get_logger(self):
        return self._logger


def spin_some(): #fire all timer that is due. once per sim frame
    for t in list(_TIMERS):
        if CLOCK.ns >= t.next_ns:
            t.next_ns += t.period_ns
            if t.next_ns <= CLOCK.ns: #fell behind: skip, do not burst-fire (bad)
                t.next_ns = CLOCK.ns + t.period_ns
            t.callback()


def reset(): #forget all topics, timers and time (reset)
    _SUBSCRIBERS.clear()
    _TIMERS.clear()
    CLOCK.ns = 0


#install
def _module(name, is_package=False, **attrs):
    m = types.ModuleType(name)
    if is_package:
        m.__path__ = []
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def install(): #register fake rclpy / *_msgs modules so `from rclpy.node import Node` etc. work
    rclpy = _module("rclpy", True,
                    init=lambda *a, **k: None,
                    shutdown=lambda *a, **k: None,
                    ok=lambda: True,
                    spin_once=lambda *a, **k: spin_some(),
                    spin_some=spin_some)
    rclpy.node = _module("rclpy.node", Node=Node) #not entirely sure how this owrks but its not core part of math and internet says works and simulation says works so :thumbs:

    for pkg, names in {
        "geometry_msgs": {"Pose2D": Pose2D},
        "std_msgs": {"Float32": Float32},
        "nav_msgs": {"OccupancyGrid": OccupancyGrid},
        "sensor_msgs": {"LaserScan": LaserScan},
        "ackermann_msgs": {"AckermannDriveStamped": AckermannDriveStamped,
                           "AckermannDrive": AckermannDrive},
    }.items():
        p = _module(pkg, True)
        p.msg = _module(pkg + ".msg", **names)