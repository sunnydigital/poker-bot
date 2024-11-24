import cv2
import numpy as np
import mss

def capture_screen():
    with mss.mss() as sct:
        # Capture the entire screen
        screenshot = sct.grab(sct.monitors[1])  # Capture the primary monitor
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert to BGR format

def get_object_location(template, frame, threshold=0.5):
    """Find object location using template matching with RGB."""
    # Perform template matching directly on RGB channels
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= threshold)
    
    # If matches are found, calculate bounding box
    if len(loc[0]) > 0:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = max_loc
        bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
        return top_left, bottom_right
    return None, None

def main():
    # Load the poker window template
    poker_table = cv2.imread('./assets/poker_table.jpg')  # Template for the poker window

    # Load button templates as RGB
    button_templates = {
        "call_button": cv2.imread('./assets/call_button.png'),
        "check_button": cv2.imread('./assets/check_button.png'),
        "fifty_percent_button": cv2.imread('./assets/fifty_percent_button.png'),
        "fold_button": cv2.imread('./assets/fold_button.png'),
        "max_button": cv2.imread('./assets/max_button.png'),
        "min_button": cv2.imread('./assets/min_button.png'),
        "minus_button": cv2.imread('./assets/minus_button.png'),
        "plus_button": cv2.imread('./assets/plus_button.png'),
        "pot_button": cv2.imread('./assets/pot_button.png'),
        "raise_to_button": cv2.imread('./assets/raise_to_button.png'),
    }

    # Dictionary to store button locations
    button_locations = {}

    while True:
        # Capture the screen
        frame = capture_screen()

        # Find the poker window
        poker_table_top_left, poker_table_bottom_right = get_object_location(poker_table, frame)

        # Loop through each button template and find its location
        for button_name, button_template in button_templates.items():
            # Skip if the template is not loaded correctly
            if button_template is None:
                print(f"Error loading template for {button_name}.")
                continue

            # Apply get_object_location
            button_top_left, button_bottom_right = get_object_location(button_template, frame)

            # Store the results in the dictionary
            button_locations[button_name] = (button_top_left, button_bottom_right)

            # Draw a rectangle around the detected button (if found)
            if button_top_left and button_bottom_right:
                cv2.rectangle(frame, button_top_left, button_bottom_right, (0, 255, 0), 2)

        print("="*70)

        # Print the poker table location
        print("Poker Table Location:", (poker_table_top_left, poker_table_bottom_right))

        # Print the detected button locations
        for button, location in button_locations.items():
            if location[0] and location[1]:
                print(f"{button}: Top-Left: {location[0]}, Bottom-Right: {location[1]}")

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()