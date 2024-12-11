import pyautogui
import time
from typing import Dict, Any, Tuple, Optional
import json
import logging
import os
from pathlib import Path
import numpy as np
import cv2
import pytesseract
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PyAutoGUI safety settings
pyautogui.FAILSAFE = True  # Moving mouse to top-left corner will abort the program
pyautogui.PAUSE = 0.5     # Delay between actions

# Button mapping configuration
BUTTON_MAPPING = {
    "fold": "fold_button.png",
    "call": "call_button.png",
    "check": "check_button.png",
    "raise": "raise_to_button.png",
    "min": "min_button.png",
    "max": "max_button.png",
    "plus": "plus_button.png",
    "minus": "minus_button.png",
    "pot": "pot_button.png",
    "fifty_percent": "fifty_percent_button.png"
}

# Button location cache
BUTTON_LOCATIONS = {name: None for name in BUTTON_MAPPING.keys()}

# Template images path
TEMPLATE_DIR = Path("assets")

def get_current_amount() -> Optional[float]:
    """
    Recognize the current bet amount displayed
    Returns:
        Optional[float]: Current amount, None if recognition fails
    """
    try:
        # Capture screenshot of the raise input area
        # Region coordinates need to be adjusted based on actual interface
        raise_region = pyautogui.screenshot(region=(
            BUTTON_LOCATIONS["raise"][0] - 100,  # x coordinate (assuming amount is displayed left of raise button)
            BUTTON_LOCATIONS["raise"][1] - 20,   # y coordinate (slight upward offset)
            200,  # width
            40    # height
        ))
        
        # Convert image format
        img_np = np.array(raise_region)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Image preprocessing
        _, img_threshold = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR recognition
        text = pytesseract.image_to_string(img_threshold, config='--psm 7 -c tessedit_char_whitelist=0123456789.')
        
        # Extract numbers
        amount = float(''.join(c for c in text if c.isdigit() or c == '.'))
        logger.info(f"Current amount: {amount}")
        return amount
        
    except Exception as e:
        logger.error(f"Error recognizing amount: {str(e)}")
        return None

def click_n_times(button: str, n: int):
    """
    Click specified button n times
    Args:
        button: Button name
        n: Number of clicks
    """
    if not BUTTON_LOCATIONS[button]:
        return
    
    for _ in range(n):
        pyautogui.click(BUTTON_LOCATIONS[button])
        time.sleep(0.1)

def adjust_raise_amount(target_amount: float) -> bool:
    """
    Adjust the raise amount
    Args:
        target_amount: Target amount to set
    Returns:
        bool: Whether adjustment was successful
    """
    try:
        # 1. Check if all necessary buttons are present
        required_buttons = ["min", "max", "plus", "minus", "pot", "fifty_percent"]
        if not all(BUTTON_LOCATIONS[btn] for btn in required_buttons):
            logger.error("Missing required raise control buttons")
            return False
        
        # 2. Try using shortcut buttons first
        current = get_current_amount()
        if current is None:
            logger.error("Unable to read current amount")
            return False
            
        # If target amount is close to half pot, use 50% button
        if abs(target_amount - current * 2) < 1:
            pyautogui.click(BUTTON_LOCATIONS["fifty_percent"])
            return True
            
        # If target amount is close to pot, use pot button
        if abs(target_amount - current) < 1:
            pyautogui.click(BUTTON_LOCATIONS["pot"])
            return True
        
        # 3. If shortcuts don't apply, use plus/minus buttons for precise adjustment
        # Start from minimum value as baseline
        pyautogui.click(BUTTON_LOCATIONS["min"])
        time.sleep(0.5)
        
        current = get_current_amount()
        if current is None:
            return False
        
        # Calculate number of adjustments needed
        step_size = 1  # Amount changed per click (adjust based on actual game)
        steps = int(abs(target_amount - current) / step_size)
        
        if target_amount > current:
            click_n_times("plus", steps)
        else:
            click_n_times("minus", steps)
        
        # Verify final amount
        final_amount = get_current_amount()
        if final_amount is None:
            return False
            
        # Allow small error margin (e.g., 0.1)
        if abs(final_amount - target_amount) <= 0.1:
            logger.info(f"Successfully adjusted to target amount: {final_amount}")
            return True
        else:
            logger.error(f"Failed to adjust to exact target amount, current: {final_amount}, target: {target_amount}")
            return False
            
    except Exception as e:
        logger.error(f"Error adjusting amount: {str(e)}")
        return False

def locate_button(button_name: str, confidence: float = 0.8) -> Optional[Tuple[int, int]]:
    """
    Locate button using template matching
    Args:
        button_name: Name of the button
        confidence: Matching confidence threshold
    Returns:
        Optional[Tuple[int, int]]: Button center coordinates, None if not found
    """
    if button_name not in BUTTON_MAPPING:
        logger.error(f"Unknown button name: {button_name}")
        return None
        
    template_path = TEMPLATE_DIR / BUTTON_MAPPING[button_name]
    
    if not template_path.exists():
        logger.error(f"Button template image not found: {template_path}")
        return None
    
    try:
        # Locate button
        button_location = pyautogui.locateCenterOnScreen(
            str(template_path),
            confidence=confidence
        )
        
        if button_location:
            return (button_location.x, button_location.y)
        else:
            logger.warning(f"Button not found: {button_name}")
            return None
            
    except Exception as e:
        logger.error(f"Error locating button {button_name}: {str(e)}")
        return None

def setup_button_locations(confidence: float = 0.8) -> bool:
    """
    Set up all button locations using template matching
    Args:
        confidence: Matching confidence threshold
    Returns:
        bool: Whether all required buttons were successfully located
    """
    # Update all button locations
    for button_name in BUTTON_MAPPING.keys():
        location = locate_button(button_name, confidence)
        if location:
            BUTTON_LOCATIONS[button_name] = location
            logger.info(f"Successfully located button {button_name}: {location}")
    
    # Check if all required buttons are found
    required_buttons = ["fold", "call", "check", "raise"]
    missing_buttons = [btn for btn in required_buttons if not BUTTON_LOCATIONS[btn]]
    
    if missing_buttons:
        logger.error(f"Missing required buttons: {missing_buttons}")
        return False
    
    return True

def execute_action(action: Dict[str, Any]) -> bool:
    """
    Execute poker action
    Args:
        action: Action information dictionary, format:
        {
            "action": "fold/call/check/raise",
            "amount": null/number,
            "confidence": 0.0-1.0,
            "reasoning": "explanation"
        }
    Returns:
        bool: Whether action was executed successfully
    """
    try:
        action_type = action["action"].lower()
        logger.info(f"Executing action: {action_type}")
        
        # Relocate buttons before each action
        if not setup_button_locations():
            logger.error("Unable to locate required buttons")
            return False
        
        if action_type == "fold":
            return execute_fold()
        elif action_type == "call":
            return execute_call()
        elif action_type == "check":
            return execute_check()
        elif action_type == "raise":
            return execute_raise(action["amount"])
        else:
            logger.error(f"Unknown action type: {action_type}")
            return False
            
    except Exception as e:
        logger.error(f"Error executing action: {str(e)}")
        return False

def execute_fold() -> bool:
    """Execute fold action"""
    try:
        if BUTTON_LOCATIONS["fold"]:
            pyautogui.click(BUTTON_LOCATIONS["fold"])
            logger.info("Executed fold")
            return True
        else:
            logger.error("Fold button not found")
            return False
    except Exception as e:
        logger.error(f"Error executing fold: {str(e)}")
        return False

def execute_call() -> bool:
    """Execute call action"""
    try:
        if BUTTON_LOCATIONS["call"]:
            pyautogui.click(BUTTON_LOCATIONS["call"])
            logger.info("Executed call")
            return True
        else:
            logger.error("Call button not found")
            return False
    except Exception as e:
        logger.error(f"Error executing call: {str(e)}")
        return False

def execute_check() -> bool:
    """Execute check action"""
    try:
        if BUTTON_LOCATIONS["check"]:
            pyautogui.click(BUTTON_LOCATIONS["check"])
            logger.info("Executed check")
            return True
        else:
            logger.error("Check button not found")
            return False
    except Exception as e:
        logger.error(f"Error executing check: {str(e)}")
        return False

def execute_raise(amount: float) -> bool:
    """
    Execute raise action
    Args:
        amount: Raise amount
    """
    try:
        # 1. Click raise button
        if not BUTTON_LOCATIONS["raise"]:
            logger.error("Raise button not found")
            return False
        
        pyautogui.click(BUTTON_LOCATIONS["raise"])
        time.sleep(0.5)
        
        # 2. Adjust amount
        if not adjust_raise_amount(amount):
            logger.error("Failed to adjust raise amount")
            return False
        
        # 3. Confirm raise
        pyautogui.click(BUTTON_LOCATIONS["raise"])
        logger.info(f"Executed raise: {amount}")
        return True
        
    except Exception as e:
        logger.error(f"Error executing raise: {str(e)}")
        return False