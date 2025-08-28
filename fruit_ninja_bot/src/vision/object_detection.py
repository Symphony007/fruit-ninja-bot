import cv2
import numpy as np

def find_targets(frame, lower_color, upper_color, min_area):
    """
    Find objects in the frame based on color masking and contour detection.
    
    Args:
        frame: The input BGR image frame from the capture.
        lower_color: Lower bound of the HSV color range.
        upper_color: Upper bound of the HSV color range.
        min_area: Minimum contour area to consider as a valid object.
        
    Returns:
        A list of dictionaries, each containing the center coordinates and area of a detected object.
        Format: [{"center": (cX, cY), "area": area}, ...]
    """
    # Convert frame from BGR to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a mask for the given color range
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)
    
    # Optional: Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_objects = []
    
    # Process each contour
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        
        # If the area is large enough, process it
        if area > min_area:
            # Calculate the center of the contour (centroid)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                detected_objects.append({
                    "center": (cX, cY),
                    "area": area
                })
    
    return detected_objects


# Optional: Helper function for debugging to visualize the mask
def show_mask_debug(frame, lower_color, upper_color, window_name="Debug Mask"):
    """
    Helper function to show the mask used for detection for debugging purposes.
    """
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)
    cv2.imshow(window_name, mask)
