import cv2
import numpy as np
import mss
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt, QRect
import sys


def capture_screen():
    with mss.mss() as sct:
        # Capture the entire screen
        screenshot = sct.grab(sct.monitors[1])  # Capture the primary monitor
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert to BGR format


def get_object_location(template, frame, threshold=0.5):
    """Find object location using template matching with RGB."""
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= threshold)

    # If matches are found, calculate bounding box
    if len(loc[0]) > 0:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = max_loc
        bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
        return top_left, bottom_right
    return None, None


class OverlayApp(QMainWindow):
    def __init__(self, poker_table, button_templates):
        super().__init__()
        self.setWindowTitle("Screen Overlay")
        self.setGeometry(0, 0, 1920, 1080)  # Adjust for your screen resolution
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.poker_table = poker_table
        self.button_templates = button_templates
        self.rectangles = []

    def update_rectangles(self):
        """Capture the screen and find object locations."""
        self.rectangles = []  # Clear previous rectangles
        frame = capture_screen()

        # Find poker table location
        poker_table_top_left, poker_table_bottom_right = get_object_location(self.poker_table, frame)
        if poker_table_top_left and poker_table_bottom_right:
            self.rectangles.append(QRect(*poker_table_top_left,
                                         poker_table_bottom_right[0] - poker_table_top_left[0],
                                         poker_table_bottom_right[1] - poker_table_top_left[1]))

        # Find button locations
        for button_name, button_template in self.button_templates.items():
            button_top_left, button_bottom_right = get_object_location(button_template, frame)
            if button_top_left and button_bottom_right:
                self.rectangles.append(QRect(*button_top_left,
                                             button_bottom_right[0] - button_top_left[0],
                                             button_bottom_right[1] - button_top_left[1]))
        self.repaint()  # Trigger a repaint with updated rectangles

    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen(Qt.green, 2)  # Green pen for rectangles
        painter.setPen(pen)

        # Draw each rectangle
        for rect in self.rectangles:
            painter.drawRect(rect)


def main():
    app = QApplication(sys.argv)

    # Load templates
    poker_table = cv2.imread('./assets/poker_table.jpg')
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

    overlay = OverlayApp(poker_table, button_templates)
    overlay.show()

    # Periodically update rectangles
    from PyQt5.QtCore import QTimer
    timer = QTimer()
    timer.timeout.connect(overlay.update_rectangles)
    timer.start(100)  # Update every 100ms

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
