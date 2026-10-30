import numpy as np

import fake_ros
fake_ros.install()  #must happen before importing the nodes (they import rclpy / *_msgs)

from se_node import StateEstimationNode
from controller_node import SLAMController
import ekf_models as m

WHEELBASE = 0.22 #m
GRID_CELLS = 700 #400 x 400 res
GRID_RES = 0.05 # of 5 cm cells


def build_nodes(goal_xy):
    fake_ros.reset()

    se = StateEstimationNode(
        initial_state=np.zeros(3),
        initial_covariance={
            'state': np.diag([1e-3, 1e-3, 1e-3]),
            'measurement': np.diag([0.03 ** 2, 0.03 ** 2, 0.02 ** 2]), #ICP pose noise
            'process': np.diag([0.05 ** 2, 0.05 ** 2]), #speed m/s, steering rad
        },
        models={'state': m.state_model, 'measurement': m.measurement_model},
        jacobians={'state_transition': m.state_transistion_jacobian,
                   'measurement': m.measurement_jacobian,
                   'process_noise': m.process_noise_jacobian},
        grid_params={'width': GRID_CELLS, 'height': GRID_CELLS, 'resolution': GRID_RES},
        grid_odds={'prior': 0.5, 'hit': 0.7, 'miss': 0.3},
        sys_params={'wheelbase': WHEELBASE},
        publish_rate=10.0,
        beam_stride=3, #1080 beams -> 360 for the map, for speedup
    )

    controller = SLAMController(
        goal_xy=goal_xy,
        lookahead_cells=8, #8*5 = 40 cm lkhd
        pid_gains=(0.6, 0.0, 0.0),
        control_rate=10.0,
        base_speed=1, #out of control 0-1 not m/s
        max_steering_angle=1, #based on cotnrol
        inflation_radius=0.35, #m, give car radius to avoid
        goal_tolerance=0.2,
    )
    return se, controller
