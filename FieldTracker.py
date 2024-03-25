import cv2
import numpy as np
from KalmanFilter import KalmanFilter  # Assuming this is your imported Kalman Filter class


KF = KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1)

def select_user_rois(frame):
    # User selects the rois in the first frame
    rois = cv2.selectROIs('Select ROIs', frame, fromCenter=False, showCrosshair=False)
    cv2.destroyWindow('Select ROIs')

    # returns a list of tuples, each representing an roi in (x, y, w, h) format
    return rois

    
def create_mask_and_hist(frame, x, y, w, h):
    # extract roi
    roi = frame[y:y+h, x:x+w]
    # show roi
    cv2.imshow('ROI', roi)

    # roi converted from BGR to the HSV (Hue, Saturation, Value) color space
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # creates a binary mask where pixels within the specified range are set to white  and all others to black
    # 0 - 180 Hue:          include all hues
    # 60 - 255 Saturation:  exclude very dull colours which could be due to lighting conditions
    # 32 - 255 Value:       filter out very dark pixels which could help deal with shadows
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

    # calculates the color histogram of the masked roi in the HSV space, but only for the Hue channel (channel ranges from 0 to 180)
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])

    # normalize so that all values fall between 0 and 255
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    return roi_hist


def track_objects(frame, roi_hists, kalman_filters, term_crit):
    """
    Tracking overview:

    1. Convert Frame to HSV: 
        Convert the current frame to the HSV color space to prepare for histogram back-projection.

    2. Histogram Back-Projection: 
        For each object (ROI) being tracked, use the HSV frame to create a probability map indicating the likelihood of each pixel belonging to the object, 
        based on the color histogram created from the initial ROI.

    3. Predict with Kalman Filter: 
        Before locating the object in the current frame, use the Kalman Filter to predict the object's new position based on its previously known state. 
        This step gives an a priori estimate of the object's current location.

    4. Apply MeanShift for Location Refinement: 
        Use the MeanShift algorithm on the probability map to refine the object's location. 
        MeanShift is guided towards areas of high probability (high color match) starting from the predicted position by the Kalman Filter.
        This results in the MeanShift algorithm finding the new location of the object.

    5. Extract New Position: 
        From the MeanShift result, extract the new bounding box (x, y, w, h) of the object.
        This represents the detected position of the object in the current frame.

    6. Draw and Display: 
        Draw a rectangle around the new location of the object on the frame for visualization and display the updated frame.
    
    7. Update Kalman Filter with Measurement: 
        Update the Kalman Filter with the actual measurement, which is the center of the new bounding box found by MeanShift. 
        This step corrects the predicted state based on the new observation, resulting in an a posteriori estimate combining the model's prediction with the observed data.
    """

    for i, roi_hist in enumerate(roi_hists):
        frame_copy = frame.copy()

        # Gaussian blur to reduce noise
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # convert frame to HSV and back-project the histogram to get a probability map
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        
        # predict the next position using the Kalman Filter
        predicted = kalman_filters[i].predict()
        predicted_x, predicted_y = int(predicted[0]), int(predicted[1])

        # apply MeanShift to find the new location based on the back-projected histogram
        ret, track_window = cv2.meanShift(dst, (predicted_x, predicted_y, roi_hist.shape[1], roi_hist.shape[0]), term_crit)

        # extract the new position (it's the MeanShift result)
        x, y, w, h = track_window

        # draw the tracked object position on the image
        img2 = cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('Tracking', img2)

        # update Kalman Filter with the new measurement
        # the measurement is the center of the tracked window
        measurement = np.array([[np.float32(x + w/2)], [np.float32(y + h/2)]])
        corrected = kalman_filters[i].update(measurement)


def main():
    filename = 'field_map.mp4'
    cap = cv2.VideoCapture(filename)

    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return

    # for the first frame the user selects the ROIs
    rois = select_user_rois(frame)
    print("Selected ROIs:", rois)
    roi_hists = [create_mask_and_hist(frame, roi[0], roi[1], roi[2], roi[3]) for roi in rois]
    kalman_filters = [KF for _ in rois]  # initialize a Kalman Filter for each ROI

    # set up the termination criteria:
    # TERM_CRITERIA_EPS: 
    #   - tells the algorithm to stop when the change in the centroid for our meanShift tracking is smaller than a specified epsilon (eps) 
    #   - It signifies convergence to a solution
    #   - we have specified 1 as our eps in this case
    # TERM_CRITERIA_COUNT: 
    #   - tells the algorithm to stop after a specified number of iterations 
    #   - we have specificed 100 as our iterations in this case
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1 )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        track_objects(frame, roi_hists, kalman_filters, term_crit)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
