import matplotlib.pyplot as plt 
import numpy as np 
import cv2

def get_line(pt1, pt2):
    """
    Calculate the line passing through two points using cross product.

    Args:
        pt1 (numpy.ndarray): First point coordinates.
        pt2 (numpy.ndarray): Second point coordinates.

    Returns:
        numpy.ndarray: Line coefficients.
    """
    line = np.cross(pt1, pt2)
    normalized_line = line / line[2]
    return normalized_line

def draw_points_and_line(img, pts, norm_pm, norm_vanish, norm_lm):
    img_copy = img.copy()

    for pt in pts:
        img_copy = cv2.circle(img_copy, (int(pt[0]), int(pt[1])), 10, [255, 255, 0], -1)

    # Draw mid point
    img_copy = cv2.circle(img_copy, (int(norm_pm[0]), int(norm_pm[1])), 10, [0, 255, 255], -1)

    # Draw vanishing point
    img_copy = cv2.circle(img_copy, (int(norm_vanish[0]), int(norm_vanish[1])), 10, [0, 0, 255], -1)
    
    img_copy = cv2.line(img_copy, (norm_pm[0], norm_pm[1]), (norm_vanish[0], norm_vanish[1]), [255, 0, 0], 2)

    return img_copy

def main(image_path, num_pts=4):
    # Load the image
    img = plt.imread(image_path)
    fig, ax = plt.subplots() 
    ax.imshow(img) 
    ax.axis('off') 
    plt.title("Image") 

    # Get user-defined points
    pts = np.array(plt.ginput(num_pts))
    pts = np.concatenate((pts, np.ones([num_pts, 1])), axis=1) # Convert the points to homogeneous coordinates.
    print(pts)
    # Calculate vanishing point
    l1 = get_line(pts[0], pts[3])
    l2 = get_line(pts[1], pts[2])
    l3 = get_line(pts[0], pts[1])
    l4 = get_line(pts[2], pts[3])
    print(l1, l2)
    
    pm = np.cross(l1, l2)
    norm_pm = (pm / pm[2]).astype(np.int32)

    vanish = np.cross(l3, l4)
    norm_vanish = (vanish / vanish[2]).astype(np.int32)

    lm = get_line(vanish, pm)
    norm_lm = (lm / lm[2]).astype(np.int32)

    
    # Draw points and line on the image
    img = draw_points_and_line(img, pts, norm_pm, norm_vanish, norm_lm)
    
    # Display the image with points and line
    cv2.imshow("Q2a", img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("../assets/tennis_court_background.png")
