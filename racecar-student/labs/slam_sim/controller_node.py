import time
import math

from rclpy.node import Node
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import OccupancyGrid
from ackermann_msgs.msg import AckermannDriveStamped

from astar import AStarPlanner

#units: meters, radians (steering: positive = l, ROS convention), m/s

class PID:  # pid class very simple
    def __init__(self, kP=0, kI=0, kD=0):
        self.kP = kP
        self.kI = kI
        self.kD = kD
        self.reset()

    def reset(self):
        self.prev_error = 0.0
        self.cum_i_val = 0.0
        self.prev_tick_called = None #prevent first dt being ~ the whole uptime (whoopsy)

    def tick(self, setpoint, val, reset=False, now=None):
        if reset:
            self.reset()
        if now is None:
            now = time.perf_counter()

        error = val - setpoint
        dt = 0.0 if self.prev_tick_called is None else now - self.prev_tick_called #dt

        p = self.kP * error #p term
        if dt > 0:
            self.cum_i_val += self.kI * error * dt #iterm
        d = self.kD * (error - self.prev_error) / dt if dt > 0 else 0.0 #dterm

        self.prev_error = error
        self.prev_tick_called = now #update for dt calcs

        return p + self.cum_i_val + d

class SLAMController(Node):
    def __init__(self, goal_xy, lookahead_cells=8, pid_gains=(1.0, 0.0, 0.0), #buncha params, configurbale
                 obstacle_threshold=60, control_rate=10.0, base_speed=0.5,
                 max_steering_angle=0.4695, inflation_radius=0.35, goal_tolerance=0.2):
        super().__init__('slam_controller')

        self.pose_sub = self.create_subscription( #read from state estimate pose, so we know where we are
            Pose2D, '/state_estimate/pose', self.pose_callback, 10
        )
        self.map_sub = self.create_subscription( #read from state estimate map, so we know where obstacles are
            OccupancyGrid, '/state_estimate/map', self.map_callback, 10
        )
        self.drive_pub = self.create_publisher( #push commands to drive topic which should control servo/motor
            AckermannDriveStamped, '/drive', 10
        )

        self.goal_xy = goal_xy #store params in self
        self.lookahead_cells = lookahead_cells
        self.obstacle_threshold = obstacle_threshold
        self.base_speed = base_speed
        self.max_steering_angle = max_steering_angle
        self.inflation_radius = inflation_radius 
        self.goal_tolerance = goal_tolerance

        kP, kI, kD = pid_gains
        self.pid = PID(kP=kP, kI=kI, kD=kD)

        self.current_pose = None
        self.current_map = None

        self.control_timer = self.create_timer( #timer to call control loop at rate
            1.0 / control_rate, self.control_loop
        )

    def pose_callback(self, data): #when i get new pose (dtype pose2d)
        self.current_pose = data #update

    def map_callback(self, data): #when i get new map (dtype occupancygrid)
        self.current_map = data #update

    def _get_lookahead_target(self, path): #walk lookahead along planned path
        return path[min(self.lookahead_cells, len(path) - 1)] #index of lookahead point, plus bounds protection

    @staticmethod
    def _world_to_body(dx, dy, theta): #helper to get to body coordinates
        forward = dx * math.cos(theta) + dy * math.sin(theta)
        left = -dx * math.sin(theta) + dy * math.cos(theta)
        return left, forward

    def control_loop(self): #main loop function that runs at hz rate
        if self.current_pose is None or self.current_map is None: #not enough data
            return

        pose = self.current_pose
        if math.hypot(self.goal_xy[0] - pose.x, self.goal_xy[1] - pose.y) < self.goal_tolerance:
            self._publish_stop()
            return

        try: #attempt to plan path
            planner = AStarPlanner( #init astar
                self.current_map,
                obstacle_threshold=self.obstacle_threshold,
                allow_unknown=True, # unknown cells treated as passable, so it tries to go trhough uknown to explore
                inflation_radius=self.inflation_radius,
            )
            path = planner.plan((pose.x, pose.y), self.goal_xy) #get path
        except ValueError as e: #path invalid, or other error
            self.get_logger().warn(f"Planning failed: {e}")
            self._publish_stop()
            return

        if path is None or len(path) < 2: #bad path
            self.get_logger().warn("No path found to goal.")
            self._publish_stop()
            return

        target_x, target_y = self._get_lookahead_target(path) #get lookahead target
        left, forward = self._world_to_body(target_x - pose.x, target_y - pose.y, pose.theta)

        heading_error = math.atan2(left, forward) #heading error (angle to target)

        now_s = self.get_clock().now().nanoseconds * 1e-9
        steering = self.pid.tick(0.0, heading_error, now=now_s) #steering based on heading error and pid
        steering = max( #clamp
            -self.max_steering_angle,
            min(self.max_steering_angle, steering)
        )

        drive_msg = AckermannDriveStamped() #create msg
        drive_msg.header.stamp = self.get_clock().now().to_msg() #stamp with time
        drive_msg.drive.steering_angle = float(steering) #set angle
        drive_msg.drive.speed = float(self.base_speed) #set speed

        self.drive_pub.publish(drive_msg) #publish; note: will use speed controller in addition in future, but just testing basics now

    def _publish_stop(self): #helper to publish a stop moving command
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.steering_angle = 0.0
        drive_msg.drive.speed = 0.0
        self.drive_pub.publish(drive_msg)
        self.pid.reset()