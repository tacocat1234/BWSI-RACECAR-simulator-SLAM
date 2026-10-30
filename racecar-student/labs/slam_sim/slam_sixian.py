import numpy as np


class SLAM: #kinda jsut ekf + occ wrapper
    def __init__(self, ekf, occupancy_grid, sys_params, beam_stride=1):
        self.ekf = ekf
        self.occupancy_grid = occupancy_grid
        self.sys_params = sys_params
        self.beam_stride = beam_stride

    def get_state_estimate(self): #getter
        return self.ekf.state_estimate.copy() #copy: callers shouldnt mutate the filter state

    def get_map(self): #cooler getter
        return self.occupancy_grid.grid

    def get_map_scale(self): #less cool getter
        return self.occupancy_grid.resolution

    def update_occupancy_grid(self, pt_cloud): #use occupancygrids self update function to update current funciton given a new point cloud from lidar
        self.occupancy_grid.update_grid(pt_cloud, self.get_state_estimate(), beam_stride=self.beam_stride)

    def estimate_callback(self, control_input, delta_t):
        self.ekf.predict_state(control_input, delta_t, **self.sys_params) #just use ekf

    def measurement_callback(self, measured_position, delta_t):
        self.ekf.update_state(measured_position, delta_t, **self.sys_params) #just use ekf