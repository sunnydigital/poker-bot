import pyautogui
import time
from typing import Dict, Any, Tuple, Optional, List
import json
import logging
import os
from pathlib import Path
import numpy as np
import cv2
from .screen_capture import capture_screen
import pytesseract

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PyAutoGUI safety settings
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.5

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

# Button groups that are mutually exclusive (cannot appear at the same time)
MUTUALLY_EXCLUSIVE_BUTTONS = [
    {"call", "check"},  # Either call or check, not both
    {"min", "max"},     # Min and max buttons should be far apart
]

# Button-specific confidence thresholds
BUTTON_CONFIDENCE = {
    "fold": 0.6,
    "call": 0.6,
    "check": 0.6,
    "raise": 0.45,  # Lower threshold for raise button
    "min": 0.6,
    "max": 0.55,    # Lower threshold for max button
    "plus": 0.55,   # Lower threshold for plus button
    "minus": 0.55,  # Lower threshold for minus button
    "pot": 0.6,
    "fifty_percent": 0.6
}

# Expected relative positions (approximate)
BUTTON_POSITION_RULES = {
    "fold": {"x_range": (0.6, 0.95), "y_range": (0.5, 0.8)},
    "call": {"x_range": (0.6, 0.95), "y_range": (0.5, 0.8)},
    "check": {"x_range": (0.6, 0.95), "y_range": (0.5, 0.8)},
    "raise": {"x_range": (0.6, 0.95), "y_range": (0.5, 0.8)},
    "min": {"x_range": (0.6, 0.95), "y_range": (0.5, 0.8)},
    "max": {"x_range": (0.6, 0.95), "y_range": (0.5, 0.8)},
    "plus": {"x_range": (0.6, 0.95), "y_range": (0.5, 0.9)},    # Expanded y range for plus
    "minus": {"x_range": (0.6, 0.95), "y_range": (0.5, 0.9)},   # Expanded y range for minus
    "pot": {"x_range": (0.6, 0.95), "y_range": (0.5, 0.8)},
    "fifty_percent": {"x_range": (0.6, 0.95), "y_range": (0.5, 0.8)}
}

# Button location cache
BUTTON_LOCATIONS = {name: None for name in BUTTON_MAPPING.keys()}

# Template images path
TEMPLATE_DIR = Path("assets")

def is_valid_button_position(button_name: str, x: int, y: int, frame_width: int, frame_height: int) -> bool:
    """
    Check if button position is valid according to predefined rules
    """
    if button_name not in BUTTON_POSITION_RULES:
        return True
        
    rules = BUTTON_POSITION_RULES[button_name]
    x_ratio = x / frame_width
    y_ratio = y / frame_height
    
    x_valid = rules["x_range"][0] <= x_ratio <= rules["x_range"][1]
    y_valid = rules["y_range"][0] <= y_ratio <= rules["y_range"][1]
    
    if not (x_valid and y_valid):
        logger.debug(f"Button {button_name} position ({x_ratio:.2f}, {y_ratio:.2f}) outside expected range")
        return False
    return True

def check_mutual_exclusion(button_name: str, x: int, y: int, detected_buttons: Dict[str, Tuple[int, int]]) -> bool:
    """
    Check if button position conflicts with already detected buttons
    """
    # Check distance to existing buttons
    MIN_BUTTON_DISTANCE = 20  # Minimum pixel distance between different buttons
    
    for existing_name, (ex, ey) in detected_buttons.items():
        if existing_name == button_name:
            continue
            
        # Check if buttons are in mutually exclusive groups
        for group in MUTUALLY_EXCLUSIVE_BUTTONS:
            if button_name in group and existing_name in group:
                distance = np.sqrt((x - ex)**2 + (y - ey)**2)
                if distance < MIN_BUTTON_DISTANCE:
                    logger.debug(f"Mutually exclusive buttons too close: {button_name} and {existing_name}")
                    return False
                
    return True

def get_object_location(template: np.ndarray, frame: np.ndarray, threshold: float = 0.6) -> List[Tuple[Tuple[int, int], Tuple[int, int], float]]:
    """
    Find all potential object locations using template matching
    Returns:
        List of (top_left, bottom_right, confidence) tuples
    """
    frame_height, frame_width = frame.shape[:2]
    results = []
    
    # Convert images to grayscale for faster processing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Try different scales
    scales = [1.0, 0.75, 0.5, 1.25]
    
    for scale in scales:
        width = int(template.shape[1] * scale)
        height = int(template.shape[0] * scale)
        
        if width >= frame_width or height >= frame_height:
            continue
            
        # Resize template
        resized_template = cv2.resize(gray_template, (width, height))
        
        # Try template matching
        try:
            result = cv2.matchTemplate(gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)
            
            for y, x in zip(*locations):
                if x + width <= frame_width and y + height <= frame_height:
                    confidence = result[y, x]
                    top_left = (x, y)
                    bottom_right = (x + width, y + height)
                    results.append((top_left, bottom_right, confidence))
                    
        except Exception as e:
            logger.debug(f"Template matching failed for scale {scale}: {str(e)}")
            continue
    
    # Sort by confidence and remove duplicates
    results.sort(key=lambda x: x[2], reverse=True)
    filtered_results = []
    for r in results:
        # Check if this result overlaps with any existing result
        is_duplicate = False
        for f in filtered_results:
            x1, y1 = r[0]
            x2, y2 = f[0]
            if abs(x1 - x2) < 10 and abs(y1 - y2) < 10:  # If centers are close
                is_duplicate = True
                break
        if not is_duplicate:
            filtered_results.append(r)
    
    return filtered_results[:5]  # Return top 5 unique results

def locate_button(button_name: str, confidence: float = 0.6, detected_buttons: Dict[str, Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
    """
    Locate button using template matching with additional validation
    """
    if detected_buttons is None:
        detected_buttons = {}
        
    if button_name not in BUTTON_MAPPING:
        logger.error(f"Unknown button name: {button_name}")
        return None
        
    # Use button-specific confidence threshold
    button_confidence = BUTTON_CONFIDENCE.get(button_name, confidence)
        
    template_path = TEMPLATE_DIR / BUTTON_MAPPING[button_name]
    if not template_path.exists():
        logger.error(f"Button template image not found: {template_path}")
        return None
    
    try:
        template = cv2.imread(str(template_path))
        if template is None:
            logger.error(f"Failed to load template image: {template_path}")
            return None
            
        screen = capture_screen()
        screen_height, screen_width = screen.shape[:2]
        
        # Get all potential matches
        matches = get_object_location(template, screen, button_confidence)
        
        # Debug: Save template and screen
        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(debug_dir / f"{button_name}_template.png"), template)
        cv2.imwrite(str(debug_dir / "current_screen.png"), screen)
        
        # Debug: Draw all potential matches
        debug_screen = screen.copy()
        for top_left, bottom_right, match_confidence in matches:
            center_x = top_left[0] + (bottom_right[0] - top_left[0]) // 2
            center_y = top_left[1] + (bottom_right[1] - top_left[1]) // 2
            x_ratio = center_x / screen_width
            y_ratio = center_y / screen_height
            
            # Draw rectangle with different colors based on validation
            color = (0, 0, 255)  # Red for invalid matches
            
            # Check position validity
            position_valid = is_valid_button_position(button_name, center_x, center_y, screen_width, screen_height)
            if position_valid:
                color = (255, 0, 0)  # Blue for position-valid matches
                
                # Check mutual exclusion
                if check_mutual_exclusion(button_name, center_x, center_y, detected_buttons):
                    color = (0, 255, 0)  # Green for fully valid matches
            
            cv2.rectangle(debug_screen, top_left, bottom_right, color, 2)
            cv2.putText(debug_screen, 
                       f"{button_name} ({match_confidence:.2f}) ({x_ratio:.2f}, {y_ratio:.2f})", 
                       (top_left[0], top_left[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imwrite(str(debug_dir / f"{button_name}_all_matches.png"), debug_screen)
        
        # Try matches in order of confidence until we find a valid one
        for top_left, bottom_right, match_confidence in matches:
            center_x = top_left[0] + (bottom_right[0] - top_left[0]) // 2
            center_y = top_left[1] + (bottom_right[1] - top_left[1]) // 2
            
            # Debug position information
            x_ratio = center_x / screen_width
            y_ratio = center_y / screen_height
            logger.debug(f"Checking match for {button_name} at ({x_ratio:.2f}, {y_ratio:.2f}) with confidence {match_confidence:.3f}")
            
            # Validate position
            if not is_valid_button_position(button_name, center_x, center_y, screen_width, screen_height):
                logger.debug(f"Position validation failed for {button_name}")
                continue
                
            # Check mutual exclusion
            if not check_mutual_exclusion(button_name, center_x, center_y, detected_buttons):
                logger.debug(f"Mutual exclusion check failed for {button_name}")
                continue
            
            # Valid match found
            logger.info(f"Found button {button_name} at ({center_x}, {center_y}) with confidence {match_confidence:.3f}")
            
            # Debug: Draw final detection
            debug_screen = screen.copy()
            cv2.rectangle(debug_screen, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(debug_screen, 
                       f"{button_name} ({match_confidence:.2f}) ({x_ratio:.2f}, {y_ratio:.2f})", 
                       (top_left[0], top_left[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(str(debug_dir / f"{button_name}_detected.png"), debug_screen)
            
            return (center_x, center_y)
            
        logger.warning(f"No valid match found for button: {button_name}")
        return None
            
    except Exception as e:
        logger.error(f"Error locating button {button_name}: {str(e)}")
        return None

def setup_button_locations(confidence: float = 0.6) -> bool:
    """
    Set up all button locations with mutual exclusion checks
    """
    detected = {}
    
    # First locate the main action buttons in specific order
    main_buttons = ["call", "fold", "raise"]  # Check call first as it's most reliable
    for button_name in main_buttons:
        location = locate_button(button_name, confidence, detected)
        if location:
            detected[button_name] = location
            BUTTON_LOCATIONS[button_name] = location
    
    # If we found call, don't look for check
    if not BUTTON_LOCATIONS.get("call"):
        location = locate_button("check", confidence, detected)
        if location:
            detected["check"] = location
            BUTTON_LOCATIONS["check"] = location
    
    # Then locate the secondary buttons
    secondary_buttons = ["min", "max", "pot", "fifty_percent", "plus", "minus"]
    for button_name in secondary_buttons:
        location = locate_button(button_name, confidence, detected)
        if location:
            detected[button_name] = location
            BUTTON_LOCATIONS[button_name] = location
    
    # Check if we have the minimum required buttons
    has_call_or_check = bool(BUTTON_LOCATIONS.get("call") or BUTTON_LOCATIONS.get("check"))
    has_fold = bool(BUTTON_LOCATIONS.get("fold"))
    has_raise = bool(BUTTON_LOCATIONS.get("raise"))
    
    if not (has_call_or_check and has_fold and has_raise):
        missing = []
        if not has_fold:
            missing.append("fold")
        if not has_raise:
            missing.append("raise")
        if not has_call_or_check:
            missing.append("call/check")
        logger.error(f"Missing required buttons: {missing}")
        return False
        
    return True

def get_current_amount() -> Optional[float]:
    """
    Recognize current displayed amount
    Returns:
        Optional[float]: Current amount, None if recognition fails
    """
    try:
        # Get screenshot of the raise input area
        screen = capture_screen()
        
        # Extract the region around the raise button
        if not BUTTON_LOCATIONS["raise"]:
            logger.error("Raise button location not found")
            return None
            
        x, y = BUTTON_LOCATIONS["raise"]
        roi = screen[y-20:y+20, x-100:x+100]  # Region of interest
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use OCR to read the number
        text = pytesseract.image_to_string(
            thresh,
            config='--psm 7 -c tessedit_char_whitelist=0123456789.'
        )
        
        # Extract numbers
        amount_str = ''.join(c for c in text if c.isdigit() or c == '.')
        if amount_str:
            amount = float(amount_str)
            logger.info(f"Current amount: {amount}")
            return amount
        else:
            logger.warning("No valid amount found in OCR result")
            return None
        
    except Exception as e:
        logger.error(f"Error getting current amount: {str(e)}")
        return None

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