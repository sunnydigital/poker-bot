import os
import json
import time
from game_state import GameState
from poker_agent import PokerAgent
from utils.screen_capture import capture_screen
from utils.action_executor import execute_action, setup_button_locations

def initialize():
    """Initialize the program"""
    print("Initializing, attempting to locate buttons...")
    if not setup_button_locations():
        print("Warning: Some buttons could not be automatically located")
        if input("Continue anyway? (y/n): ").lower() != 'y':
            return False
    return True

def main():
    # Initialize
    if not initialize():
        print("Initialization failed, program exiting")
        return
    
    # Create game state and AI agent
    game_state = GameState()
    poker_agent = PokerAgent()
    
    print("Program started, press Ctrl+C to exit")
    try:
        while True:
            # 1. Capture screen
            screen = capture_screen()
            
            # 2. Update game state using LLM
            game_state.update_from_screen(screen)
            
            # 3. Get AI decision
            action = poker_agent.get_action(game_state)
            
            # 4. Execute action
            if not execute_action(action):
                print("Action execution failed, please check if buttons are visible")
                if input("Retry? (y/n): ").lower() == 'y':
                    continue
                else:
                    break
            
            # 5. Wait before next operation
            time.sleep(2)  # Adjust delay as needed
            
    except KeyboardInterrupt:
        print("\nProgram stopped")

if __name__ == "__main__":
    main()