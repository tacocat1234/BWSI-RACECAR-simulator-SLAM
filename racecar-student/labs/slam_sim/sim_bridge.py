#fakes all  hardware. al in one cuz thats easier
import math
import numpy as np

import fake_ros
from fake_ros import Node, spin_some, CLOCK
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

#sim consts
LIDAR_RANGE_MIN = 0.05 #m
LIDAR_RANGE_MAX = 12.0 #m
LIDAR_PERIOD = 0.1 #sec

class RacecarBridgeNode(Node):
    def __init__(self, rc, lidar_period=LIDAR_PERIOD):
        super().__init__('racecar_bridge')
        self.rc = rc
        self.lidar_period = lidar_period

        self.encoder_pub = self.create_publisher(Float32, '/encoder/speed', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.drive_sub = self.create_subscription(AckermannDriveStamped, '/drive', self.drive_callback, 10)

        self.cmd_speed = 0.0 #m/s
        self.cmd_steer = 0.0 #rad, left positive
        self._last_scan_time = -1e9
        self._last_raw = None

        self ._max_speed = 0.25

        rc.drive.set_max_speed(1) #since this is coser to irl implementaiton it hink

    def set_max_speed(self, max_speed: float = 0.25) -> None: #stolen directly from racecar library real drive
        assert (
            0.0 <= max_speed <= 1.0
        ), f"max_speed [{max_speed}] must be between 0.0 and 1.0 inclusive."

        self._max_speed = max_speed

    #/drive -> remembered, applied at end of frame
    def drive_callback(self, msg):
        self.cmd_speed = float(msg.drive.speed)
        self.cmd_steer = float(msg.drive.steering_angle)

    def apply_drive(self):
        speed_ratio = self.cmd_speed * self._max_speed #set speed angle is like this for real robot as well
        angle_ratio = -self.cmd_steer #flip sign to convert from ros convetion to RACECAR convention
        speed_ratio = max(-1.0, min(1.0, speed_ratio))
        angle_ratio = max(-1.0, min(1.0, angle_ratio))
        self.rc.drive.set_speed_angle(speed_ratio, angle_ratio)

    def publish_encoder(self):
        msg = Float32()
        msg.data = float(self.rc.physics.get_encoder_speed()) #already m/s, forward positive
        self.encoder_pub.publish(msg)

    def publish_scan(self): #simulates output of REAL lidar not racecar lidar
        raw = np.asarray(self.rc.lidar.get_samples(), dtype=np.float32) #cm, clockwise, 0 = straight ahead
        if self._last_raw is not None and np.array_equal(raw, self._last_raw):
            return #lidar not new scan
        self._last_raw = raw.copy()

        n = len(raw)
        inc = 2.0 * math.pi / n
        cm = raw[(-np.arange(n)) % n] #math bullshitery to convert from RACECAR to ROS convention (again)
        ranges = cm / 100.0 #RACECAR reports in cm, real car produces m
        ranges[cm <= 0] = np.inf #0 is inf in RACECAR but real lidar produces inf (i think)

        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'laser'
        msg.angle_min = 0.0
        msg.angle_max = 2.0 * math.pi - inc
        msg.angle_increment = inc
        msg.range_min = LIDAR_RANGE_MIN
        msg.range_max = LIDAR_RANGE_MAX
        msg.ranges = ranges
        self.scan_pub.publish(msg)

    def step(self): #one frame
        CLOCK.advance(self.rc.get_delta_time())
        now = CLOCK.ns * 1e-9

        self.publish_encoder()
        if now - self._last_scan_time >= self.lidar_period:
            self._last_scan_time = now
            self.publish_scan()

        spin_some()
        self.apply_drive()
