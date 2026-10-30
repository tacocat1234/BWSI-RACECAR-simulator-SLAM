import numpy as np

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

#unitless since this is generic
class ExtendedKalmanFilter:
    def __init__(self, #initialization
                 initial_state_estimate,
                 initial_state_covariance,
                 process_noise_covariance,
                 measurement_noise_covariance,
                 state_transition_jacobian,
                 measurement_jacobian,
                 process_noise_jacobian,
                 state_model,
                 measurement_model,
                 angle_indices=(2,)): #last 5 are funcs

        #float copy cuz int array truncates in update_state
        self.state_estimate = np.array(initial_state_estimate, dtype=float) #np arr [x, y, theta]
        self.state_estimation_covariance = np.array(initial_state_covariance, dtype=float)
        self.process_noise_covariance = np.asarray(process_noise_covariance, dtype=float)
        self.measurement_noise_covariance = np.asarray(measurement_noise_covariance, dtype=float)

        self.state_transition_jacobian = state_transition_jacobian
        self.measurement_jacobian = measurement_jacobian
        self.process_noise_jacobian = process_noise_jacobian

        self.state_model = state_model
        self.measurement_model = measurement_model

        self.angle_indices = tuple(angle_indices)
        self.kalman_gain = None

    def _wrap(self, vec): #helper
        for i in self.angle_indices:
            vec[i] = wrap_angle(vec[i])
        return vec

    def predict_state(self, control_input, delta_t, **kwargs): #encoder update
        prior = self.state_estimate #wrong one before

        F = self.state_transition_jacobian(prior, control_input, delta_t, **kwargs) #get all jacobians and covariances wiht nice letter names
        G = self.process_noise_jacobian(prior, control_input, delta_t, **kwargs)
        P = self.state_estimation_covariance
        Q = self.process_noise_covariance

        self.state_estimate = self._wrap(np.array(self.state_model(prior, control_input, delta_t, **kwargs), dtype=float)) #update predicted state based on state model
        self.state_estimation_covariance = F @ P @ F.T + G @ Q @ G.T #update state estimation covariance by adding the uncerainty due to state (state trans jacobian) to the current coviarnace, + uncertainty due to processs noise given the current encoder measurement and how that tranlsatates to state noise (given by jacobian)

    def update_state(self, true_measurement, delta_t, **kwargs): #measurement step, compares predicted measurement to true measurement, finds difference, applies correction
        predicted_measurement = self.measurement_model(self.state_estimate, delta_t, **kwargs)
        innovation = np.asarray(true_measurement, dtype=float) - predicted_measurement #difference
        self._wrap(innovation) #359 vs 1 is err=2, not 358

        H = self.measurement_jacobian(self.state_estimate, delta_t, **kwargs) #get jacobian
        P = self.state_estimation_covariance
        R = self.measurement_noise_covariance
        I = np.identity(len(self.state_estimate))

        S = H @ P @ H.T + R #innovation covariance: basically, hpht is uncertainty of state translated into uncertainty of measurement, R, is uncertainty due to sensor noise, sum is "measurement error" (not exactly, close enough)
        self.kalman_gain = np.linalg.solve(S.T, H @ P).T #translates measurement error into a state correction; equivalent to P @ H.T @ inv(S)
        self.state_estimate = self._wrap(self.state_estimate + self.kalman_gain @ innovation) #equiv to P @ H.T @ np.linalg.inv(S) but faster or smtg idk

        L = I - self.kalman_gain @ H #helper for covariance update, basically "how much of the state is left after the measurement update"
        self.state_estimation_covariance = L @ P @ L.T + self.kalman_gain @ R @ self.kalman_gain.T #joseph form,also equiv to formulas but better cuz invariance (?)