import os
import json
import time
from datetime import datetime
from pathlib import Path
import logging
from game_state import GameState
from poker_agent import PokerAgent
from utils.screen_capture import capture_screen
from utils.action_executor import execute_action, setup_button_locations

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Main logger for program flow
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"poker_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Game results logger
results_logger = logging.getLogger('game_results')
results_logger.setLevel(logging.INFO)
results_handler = logging.FileHandler(log_dir / f"game_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
results_logger.addHandler(results_handler)
results_logger.propagate = False  # Don't propagate to root logger

# Write CSV header
results_logger.info("timestamp,hand_id,hole_cards,community_cards,pot_size,action_taken,action_amount,result,stack_change")

def initialize():
    """Initialize the program"""
    logger.info("Initializing poker bot...")
    if not setup_button_locations():
        logger.warning("Some buttons could not be automatically located")
        if input("Continue anyway? (y/n): ").lower() != 'y':
            return False
    return True

def log_game_result(game_state, action_taken, action_amount, result, stack_change):
    """Log game result to CSV file"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results_logger.info(f"{timestamp},{game_state.hand_id},{game_state.hole_cards},"
                       f"{game_state.community_cards},{game_state.pot_size},{action_taken},"
                       f"{action_amount},{result},{stack_change}")

def main():
    # Initialize
    if not initialize():
        logger.error("Initialization failed, program exiting")
        return
    
    # Create game state and AI agent
    game_state = GameState()
    poker_agent = PokerAgent()
    
    logger.info("Program started, press Ctrl+C to exit")
    try:
        while True:
            # 1. Check for available buttons first
            if not setup_button_locations():
                time.sleep(1)  # Short delay before next check
                continue
            
            # 2. Capture screen only when buttons are found
            screen = capture_screen()
            
            # 3. Update game state using LLM
            prev_state = game_state.copy()  # Store previous state to detect hand completion
            game_state.update_from_screen(screen)
            
            # Check if a new hand has started
            if game_state.hand_id != prev_state.hand_id:
                if prev_state.hand_id:  # Not the first hand
                    # Log previous hand result
                    log_game_result(
                        prev_state,
                        prev_state.last_action,
                        prev_state.last_action_amount,
                        prev_state.result,
                        prev_state.stack_change
                    )
                logger.info(f"New hand started: {game_state.hand_id}")
            
            # 4. Get AI decision
            action = poker_agent.get_action(game_state)
            
            # 5. Execute action
            if not execute_action(action):
                logger.error("Action execution failed")
                if input("Retry? (y/n): ").lower() == 'y':
                    continue
                else:
                    break
            
            # Store action for result logging
            game_state.last_action = action["action"]
            game_state.last_action_amount = action.get("amount")
            
            # 6. Wait before next operation
            time.sleep(2)  # Adjust delay as needed
            
    except KeyboardInterrupt:
        # Log final hand result if available
        if game_state.hand_id:
            log_game_result(
                game_state,
                game_state.last_action,
                game_state.last_action_amount,
                game_state.result,
                game_state.stack_change
            )
        logger.info("Program stopped by user")

if __name__ == "__main__":
    main()