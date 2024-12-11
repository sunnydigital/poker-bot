import numpy as np
import cv2
import mss

def capture_screen() -> np.ndarray:
    """
    Capture current screen state
    Returns:
        np.ndarray: Screenshot as BGR format numpy array
    """
    with mss.mss() as sct:
        # Capture the primary monitor
        screenshot = sct.grab(sct.monitors[1])
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert to BGR format for OpenCV