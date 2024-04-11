import numpy as np


class KalmanFilter:
    def __init__(self, dt, state_transition_matrix, measurement_matrix, process_noise_cov, measurement_noise_cov, error_cov_post, state_post):
        self.dt = dt                        # Time step
        self.Phi = state_transition_matrix  # State Transition Matrix
        self.H = measurement_matrix         # Measurement Matrix
        self.Q = process_noise_cov          # Process Noise Covariance
        self.R = measurement_noise_cov      # Measurement Noise Covariance
        self.P = error_cov_post             # Error Covariance Posterior
        self.x = state_post                 # State Posterior

    def predict(self):
        # state prediction
        self.x = np.dot(self.Phi, self.x)

        # error covariance
        self.P = np.dot(np.dot(self.Phi, self.P), self.Phi.T) + self.Q

        return self.x

    def update(self, z):
        # Kalman gain matrix
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(np.dot(self.H, np.dot(self.P, self.H.T)) + self.R)))

        # state updating
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))

        # error covariance updating
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))

        return self.x
    


class AdaptiveKalmanFilter(KalmanFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store default/base Q and R matrices for resetting or reference
        self.default_Q = np.copy(self.Q)
        self.default_R = np.copy(self.R)

    def adjust_for_occlusion(self, occlusion_status):
        if occlusion_status:
            # Adjust parameters to rely more on the model predictions during occlusion
            self.Q *= 0.1  # Increase model confidence
            self.R *= 1e6  # Decrease measurement confidence
        else:
            # Reset parameters to default when no occlusion is detected
            self.Q = np.copy(self.default_Q)
            self.R = np.copy(self.default_R)

    def dynamic_adjustment(self, confidence_level):
        # Adjusts the filter's parameters based on the detection confidence level.
       
        # Scale the Q and R matrices based on confidence. Lower confidence increases process noise, implying reliance on model.
        alpha_scale = max(0.1, min(confidence_level, 1))
        self.Q = self.default_Q * (1 / alpha_scale)
        self.R = self.default_R * alpha_scale