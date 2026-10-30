from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import OccupancyGrid
import numpy as np

from ekf import ExtendedKalmanFilter 
from slam_sixian import SLAM
from occupancy_grid import OccupancyGrid as OG
from slam_icp import ICPScanMatcher
from ekf_models import steering_model, config as model_config

#units: m (lidar), m/s (encoder), rad (steering, left positive)

MAX_DT = 1.0 #s; a longer gap -> assume the sim was paused / restarted, so do not integrate over it

def laserscan_to_points(scan_msg: LaserScan) -> np.ndarray: #helper
    ranges = np.asarray(scan_msg.ranges, dtype=np.float32)
    angles = scan_msg.angle_min + np.arange(len(ranges)) * scan_msg.angle_increment #compute the angle for every beam

    valid = (np.isfinite(ranges) #leep only valid measurements
             & (ranges >= scan_msg.range_min)
             & (ranges <= scan_msg.range_max))

    ranges = ranges[valid]
    angles = angles[valid]

    return np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))

class StateEstimationNode(Node):
    def __init__(self, initial_state, initial_covariance, models, jacobians, grid_params, grid_odds,
                 sys_params, publish_rate=10.0, grid_origin=None, beam_stride=3):
        super().__init__('state_estimator')

        self.measure_sub = self.create_subscription( #sub to lidar, for measure update (acucate and lsow)
            LaserScan,
            '/scan',
            self.measure_callback,
            10
        )

        self.encoder_sub = self.create_subscription( #sub to encoder, for measures
            Float32,
            '/encoder/speed',
            self.estimate_callback,
            10
        )

        self.servo_sub = self.create_subscription( #sub to drive, to read servo steering angle. this model assumes servo achives angle instantenly
            AckermannDriveStamped,
            '/drive',
            self.servo_callback,
            10
        )

        self.pose_pub = self.create_publisher( #topic to publish pose est
            Pose2D,
            '/state_estimate/pose', 
            10
        )
        self.map_pub = self.create_publisher( #topic to publish map est
            OccupancyGrid, 
            '/state_estimate/map', 
            10
        )

        sc = initial_covariance['state'] #little less stupid than unpack
        mc = initial_covariance['measurement']
        pc = initial_covariance['process']

        ekf = ExtendedKalmanFilter( #really need to stop with crappy comp prog var names
            initial_state, sc, pc, mc,
            jacobians['state_transition'], jacobians['measurement'], jacobians['process_noise'],
            models['state'], models['measurement']
        )

        ox, oy = (None, None) if grid_origin is None else grid_origin
        occ = OG(grid_params['width'], grid_params['height'], grid_params['resolution'], grid_odds['prior'], grid_odds['hit'], grid_odds['miss'], origin_x=ox, origin_y=oy) #create occupancy

        self.slam = SLAM(ekf, occ, sys_params, beam_stride=beam_stride) #create slam with the ekf and occgrid, as well as sysparams (wheel base etc)
        self.icp = ICPScanMatcher() #create icp, used to turn lidar data to useful measurement model (abs pose)

        self.commanded_angle = 0.0
        self.actual_angle = 0.0
        self.steering_rate = model_config['steering_rate'] #fetch

        self.last_measure_ns = self.get_clock().now().nanoseconds #for dt calculations 
        self.last_estimate_ns = self.last_measure_ns

        self.publish_timer = self.create_timer( #publishtimer that calls publishing at rate
            1.0 / publish_rate,
            self.publish_state_estimate
        )

    def initialize(self): #reintialize everything 
        now = self.get_clock().now().nanoseconds
        self.last_measure_ns = now
        self.last_estimate_ns = now

    def _dt(self, attr):
        now = self.get_clock().now().nanoseconds
        dt = (now - getattr(self, attr)) * 1e-9
        setattr(self, attr, now)
        return dt

    def measure_callback(self, data): #run when lidar measurement (accurate) comes in
        dt = self._dt('last_measure_ns')

        point_cloud = laserscan_to_points(data) #turn to useful pointcloud instaed of noisy lidar data

        pose = self.icp.update(point_cloud) #get icp pose 

        self.slam.measurement_callback( #update everything based on measurement (math handled in slam class)
            pose.to_array(),
            dt,
        )

        self.slam.update_occupancy_grid(point_cloud) #use pointcloud to update, as well as current updated pose

    def estimate_callback(self, data): #run when encoder data comes in
        dt = self._dt('last_estimate_ns')
        if dt <= 0 or dt > MAX_DT:
            return

       #new stuff accounts for servo max spd cuz i can; wheels move towards the commanded angle at a limited rate
        step = self.steering_rate * dt
        self.actual_angle += max(-step, min(step, self.commanded_angle - self.actual_angle)) #get true

        v = float(data.data)
        self.slam.estimate_callback( #slam runs its estiamte callback
            (v, self.actual_angle),
            dt,
        )

    def servo_callback(self, data): #store servo steering when gets
        self.commanded_angle = steering_model(float(data.drive.steering_angle))

    def get_map_estimate(self): #ros2 occgrid getter
        return self.slam.occupancy_grid.to_ros_occupancy_grid(
            frame_id="map", stamp=self.get_clock().now().to_msg())

    def get_state_estimate(self): #ros2 se getter
        state = self.slam.get_state_estimate()
        pose_msg = Pose2D()
        pose_msg.x = float(state[0])
        pose_msg.y = float(state[1])
        pose_msg.theta = float(state[2])
        return pose_msg

    def publish_state_estimate(self): #publish both pose map, callback that is on the clock
        self.pose_pub.publish(self.get_state_estimate())
        self.map_pub.publish(self.get_map_estimate())
