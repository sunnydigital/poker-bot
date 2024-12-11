import numpy as np
import cv2
import mss
import logging

logger = logging.getLogger(__name__)

def capture_screen() -> np.ndarray:
    """
    Capture current screen state
    Returns:
        np.ndarray: Screenshot as BGR format numpy array
    """
    with mss.mss() as sct:
        # Get monitor information
        monitor = sct.monitors[1]  # Primary monitor
        logger.info(f"Monitor information: {monitor}")
        
        # Calculate the actual screen dimensions
        monitor_width = monitor["width"]
        monitor_height = monitor["height"]
        
        # Capture the screen
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        
        # Convert to BGR format for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # Check if we need to handle Retina display scaling
        if frame.shape[1] > monitor_width or frame.shape[0] > monitor_height:
            scale_factor = min(monitor_width / frame.shape[1], monitor_height / frame.shape[0])
            logger.info(f"Detected Retina display, applying scale factor: {scale_factor}")
            
            # Resize the frame to match the actual screen dimensions
            frame = cv2.resize(frame, (monitor_width, monitor_height))
        
        logger.info(f"Captured frame size: {frame.shape}")
        return frame