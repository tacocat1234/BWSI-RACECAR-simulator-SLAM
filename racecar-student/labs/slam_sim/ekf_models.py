import math
import os
import numpy as np

#units of this file: x, y in meters, theta in rads, v in m/s, sigma in rads, positive = left
#state = [x, y, theta] of the car (bicycle model, reference point = rear axle (i think)).

_DEFAULTS = { #just in case
    'wheelbase': 0.22,
    'max_steering_angle': 1,
    'steering_gain': 0.4695,
    'steering_rate': math.pi,
}

def _load_config(): #better loader
    cfg = dict(_DEFAULTS)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_config.yaml') #load config from current file's directory
    try:
        import yaml
        with open(path, 'r') as f:
            cfg.update(yaml.safe_load(f) or {})
    except (ImportError, FileNotFoundError):
        pass #fall back to the defaults above
    return cfg

config = _load_config()

#all jacobians are specific to this particular racecar/design
def state_transistion_jacobian(state_estimate, control_input, delta_t, **kwargs): #x, y, heading | speed, steering angle | delta_t
    x, y, theta = state_estimate
    v, sigma = control_input

    return np.array([ #partial derivatives of the control model, or how a small change in current state affects the next state
        [1, 0, -v * math.sin(theta) * delta_t],
        [0, 1, v * math.cos(theta) * delta_t],
        [0, 0, 1],
    ])

def measurement_jacobian(state_estimate, delta_t, **kwargs): #measurement does not need control input, since unrelated
    return np.array([ #since measurement = state estimation from lidar icp directly gives pose transform, 1:1 mapping for all x, y, omega
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1] #how measurement changes when state changes. since our icp gives abs world pose, changing our pose by dx, dy, dtheta changes the measurement by the same amount, so 1:1 mapping
    ])

def process_noise_jacobian(state_estimate, control_input, delta_t, **kwargs): #x, y, heading | speed, steering angle | delta_t | wheelbase
    x, y, theta = state_estimate #noise enters through controls exclusively in this model
    v, sigma = control_input
    L = kwargs.get('wheelbase', config['wheelbase'])

    return np.array([
        [math.cos(theta) * delta_t, 0], #derivative with respect to v and sigma
        [math.sin(theta) * delta_t, 0], #pose with respect to v in column 0, pose with respect to sigma in column 1
        [math.tan(sigma) / L * delta_t, (v / L) / (math.cos(sigma) ** 2) * delta_t],
    ])

def state_model(state_estimate, control_input, delta_t, **kwargs): #copy of simplified state model from lecture 3
    x, y, theta = state_estimate
    v, sigma = control_input
    L = kwargs.get('wheelbase', config['wheelbase']) #wheelbase
    return np.array([
        x + v * math.cos(theta) * delta_t,
        y + v * math.sin(theta) * delta_t,
        theta + v / L * math.tan(sigma) * delta_t,
    ])

def measurement_model(state_estimate, delta_time, **kwargs): #maps state to predicted measurement, again 1:1 because ICP gives pose
    return np.asarray(state_estimate[:3], dtype=float) #:3

#alias with the correct spelling cuz i fucked it up the first tiem and im scared to change it and blow the whoel thingup
state_transition_jacobian = state_transistion_jacobian

def steering_model(control_angle): #convert from 0-1 control to rads
    lim = config['max_steering_angle']
    return config['steering_gain'] * max(-lim, min(lim, control_angle))