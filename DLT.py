import cv2
import numpy as np
import os

def select_points(image, num_points):
    """
    Function to allow the user to select points on an image.
    """
    points = []
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Image', image)

    cv2.imshow('Image', image)
    cv2.setMouseCallback('Image', mouse_callback)

    while len(points) < num_points:
        cv2.waitKey(100)

    cv2.destroyAllWindows()
    return np.array(points)

def estimate_homography(pts1, pts2):
    """
    Estimate homography matrix using Direct Linear Transform (DLT) algorithm.
    """
    A = []
    # Compute Ai
    for i in range(pts1.shape[0]):
        x, y = pts1[i]
        xp, yp = pts2[i]
        A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
    # Concat A
    A = np.array(A)

    # Obtain SVD of A
    _, _, V = np.linalg.svd(A)

    # Reshape
    H = V[-1].reshape(3, 3)
    return H / np.linalg.norm(H)

def apply_homography(image, H):
    """
    Apply homography to an image.
    """
    rows, cols = image.shape[:2]
    warped_image = cv2.warpPerspective(image, H, (cols, rows))
    return warped_image

if __name__ == '__main__':
    # Load images
    cap = cv2.VideoCapture("video_1.mp4")
    _, image1 = cap.read()
    image1 = cv2.resize(image1, (image1.shape[1]//2, image1.shape[0]//2))
    image2 = cv2.imread('./assets/tennis_court_background.png')
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # Select corresponding points
    num_points = 4
    pts1 = select_points(image1.copy(), num_points)
    pts2 = select_points(image2.copy(), num_points)

    # Estimate homography
    H = estimate_homography(pts1, pts2)

    # Apply homography to the first image
    warped_image = apply_homography(image1, H)

    # Display results
    cv2.imshow('Image 1', image1)
    cv2.imshow('Image 2', image2)
    cv2.imshow('Warped Image', warped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
