import numpy as np
from PIL import ImageGrab

def capture_screen():
    """
    Capture current screen state
    Returns:
        numpy.ndarray: Screenshot as numpy array
    """
    # Capture full screen using PIL's ImageGrab
    screenshot = ImageGrab.grab()
    
    # Convert PIL image to numpy array
    screen_np = np.array(screenshot)
    
    return screen_np 