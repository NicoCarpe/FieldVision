import numpy as np
import cv2


class MeanShiftTracker:
    def __init__(self, frame, roi, initial_scale=1.0):
        # Initialize the tracker
        self.frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.roi = np.array(roi)
        self.scale = initial_scale
        self.target_model = self.compute_histogram(self.frame_hsv, self.roi)
        self.kernel = self.epanechnikov_kernel(self.roi[2], self.roi[3])
        self.likelihood_img = None

    def compute_histogram(self, frame_hsv, roi, use_kernel=False):
        # Extract the specified region from the frame.
        x, y, w, h = roi 
        hsv_roi = frame_hsv[y:y+h, x:x+w]

        # Initialize mask
        mask = cv2.inRange( hsv_roi,
                            np.array((0., 0., 0.)),
                            np.array((180., 255., 255.)))
        
        if use_kernel:
            # Generate a spatial weight mask using the Epanechnikov kernel.
            kernel = self.epanechnikov_kernel(w, h)

            # use kernel as a mask
            mask = kernel.astype(np.uint8)  

        # Compute the HSV histogram of the region
        roi_hist = cv2.calcHist([hsv_roi],
                                [0, 1],
                                mask,
                                [30, 40],
                                [0, 180, 0, 256])

        # Normalize the histogram to ensure values lie within the specified range.
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        return roi_hist


    def epanechnikov_kernel(self, w, h):
        # Generate coordinate grids for kernel calculation, centered at (0,0).
        cols = np.linspace(-w / 2, w / 2, w)
        rows = np.linspace(-h / 2, h / 2, h)
        x, y = np.meshgrid(cols, rows)

        # Calculate normalized distances to center, forming a circular kernel shape.
        norm = np.sqrt(x**2 + y**2) / np.sqrt((w / 2)**2 + (h / 2)**2)

        # Apply the Epanechnikov profile to distances, zeroing out values outside the kernel radius.
        kernel = np.where(norm <= 1, 0.75 * (1 - norm**2), 0)

        # Normalize kernel weights to sum to 1, maintaining histogram integrity.
        kernel /= np.sum(kernel)

        return kernel


    def bhattacharyya_coefficient(self, hist1, hist2):
        # Calculate the Bhattacharyya coefficient for two histograms, measuring their overlap (similarity).
        return np.sum(np.sqrt(hist1 * hist2))


    def histogram_back_projection(self, current_frame, roi_hist):
        # Convert the current frame to HSV for histogram comparison.
        hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

        # Apply back-projection to map model histogram onto current frame, indicating likelihood of each pixel.
        return cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 255], 1)


    def compute_scale_adapted_kernel(self, scale_factor):
        # Recalculate kernel size based on current scale factor to adjust for target size changes.
        adapted_w = int(self.roi[2] * scale_factor)
        adapted_h = int(self.roi[3] * scale_factor)

        return self.epanechnikov_kernel(adapted_w, adapted_h), adapted_w, adapted_h


    def scale_adaptation(self, current_frame):
        # Test different scales around the current scale to find the best match for the target size.
        best_scale = self.scale

        # Initialize max similarity measure.
        max_rho = -np.inf  

        # might need to test different scale factors
        for scale in [0.85, 0.9, 1.0, 1.1, 1.15]:  
            adapted_kernel, adapted_w, adapted_h = self.compute_scale_adapted_kernel(scale)
            temp_bbox = [self.roi[0], self.roi[1], adapted_w, adapted_h]
            adapted_hist = self.compute_histogram(current_frame, temp_bbox, use_kernel=True)
            rho = self.bhattacharyya_coefficient(self.target_model, adapted_hist)
            
            # Update best scale if current scale yields a higher similarity.
            if rho > max_rho:
                max_rho = rho
                best_scale = scale

        # Update scale with a blend of current and best scale for gradual adaptation.
        self.scale = (self.scale + best_scale) / 2


    def compute_mean_shift_vector(self, likelihood_img, region):
        x, y, w, h = region

        # Ensure the ROI dimensions are positive
        if w <= 0 or h <= 0:
            return 0, 0

        roi = likelihood_img[y:y+h, x:x+w]

        # If ROI is empty after adjustments, return no movement
        if roi.size == 0:
            return 0, 0

        coord_x, coord_y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Ensure kernel_weights matches the dimensions of roi
        kernel_weights = self.epanechnikov_kernel(w, h)[:roi.shape[0], :roi.shape[1]]

        weighted_sum = np.sum(roi * kernel_weights)
        if weighted_sum == 0:
            weighted_sum = 1e-10  # Prevent division by zero

        x_mean = np.sum(coord_x * roi * kernel_weights) / weighted_sum
        y_mean = np.sum(coord_y * roi * kernel_weights) / weighted_sum

        dx = x_mean - w / 2
        dy = y_mean - h / 2

        return dx, dy


    def update_target_model(self, current_frame, update_rate=0.1):
        # Dynamically update the target model histogram to adapt to appearance changes.
        frame_hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        new_hist = self.compute_histogram(frame_hsv, self.roi, use_kernel=True)

        # Blend the new histogram with the existing model according to the update rate.
        self.target_model = (1 - update_rate) * self.target_model + update_rate * new_hist
        cv2.normalize(self.target_model, self.target_model, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


    def mean_shift(self, current_frame, max_iter=10, epsilon=1, model_update_interval=5, sim_thresh=0.3):
        # Main mean shift algorithm to locate the target within the current frame.
        frame_hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

        for i in range(max_iter):
            self.likelihood_img = self.histogram_back_projection(frame_hsv, self.target_model)
            dx, dy = self.compute_mean_shift_vector(self.likelihood_img, self.roi)

            # Update search window position based on mean shift vector.
            self.roi[0] += int(np.round(dx))
            self.roi[1] += int(np.round(dy))  

            if np.sqrt(dx**2 + dy**2) < epsilon:
                break  # Convergence check

            # Periodic target model update to account for changes in appearance.
            if i % model_update_interval == 0:
                self.update_target_model(current_frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):  # Wait for 'q' to quit
                break

        # Scale adaptation at the end of the mean shift process.
        self.scale_adaptation(current_frame)

