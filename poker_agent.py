from typing import Dict, Any
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"poker_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class PokerAgent:
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize poker AI agent
        Args:
            model_name: LLM model name to use
        """
        self.model = model_name
        self.client = OpenAI(api_key=os.getenv("GPT_API_KEY"))
        logger.info(f"Initialized PokerAgent with model: {model_name}")
        
    def get_action(self, game_state) -> Dict[str, Any]:
        """
        Get action decision based on current game state
        Args:
            game_state: GameState instance containing current game state
        Returns:
            Dict: Dictionary containing action decision
        """
        # Get current available buttons
        from utils.action_executor import BUTTON_LOCATIONS
        available_buttons = [btn for btn, loc in BUTTON_LOCATIONS.items() if loc is not None]
        
        # If no buttons are available, return a wait action
        if not available_buttons:
            logger.warning("No buttons currently available")
            return {
                "action": "wait",
                "amount": None,
                "confidence": 0.0,
                "reasoning": "No action buttons are currently available, waiting for buttons to appear"
            }
        
        # Log current game state and available buttons
        logger.info("Current Game State:")
        logger.info(f"Community Cards: {game_state.community_cards}")
        logger.info(f"Hole Cards: {game_state.hole_cards}")
        logger.info(f"Pot Size: {game_state.pot_size}")
        logger.info(f"Previous Actions: {game_state.who_raised}")
        logger.info(f"Available Buttons: {available_buttons}")
        
        # Build prompt
        prompt = self._build_prompt(game_state, available_buttons)
        logger.debug(f"Generated prompt:\n{prompt}")
        
        # Get LLM response
        response = self._get_llm_response(prompt)
        logger.info(f"Raw LLM response:\n{response}")
        
        # Parse LLM's decision
        decision = self._parse_response(response, available_buttons)
        logger.info(f"Parsed decision: {json.dumps(decision, indent=2)}")
        
        return decision
    
    def _build_prompt(self, game_state, available_buttons: list) -> str:
        """Build prompt for LLM"""
        # Convert available buttons to a more readable format
        action_options = []
        if "fold" in available_buttons:
            action_options.append("1. Fold - if the hand is weak and not worth playing")
        if "check" in available_buttons:
            action_options.append("2. Check - if you want to stay in the hand without betting")
        if "call" in available_buttons:
            action_options.append("3. Call - if the hand has potential and worth matching the current bet")
        if "raise" in available_buttons:
            action_options.append("4. Raise (specify amount) - if you have a strong hand or good bluffing opportunity")
        
        action_choices = "/".join([btn for btn in available_buttons])
        
        prompt = f"""
You are a professional poker player. Analyze the current game state and make a strategic decision based on the available actions and GTO strategy.

Current game state:
- Community Cards: {game_state.community_cards}
- Your Hole Cards: {game_state.hole_cards}
- Current Pot Size: {game_state.pot_size}
- Previous Actions: {game_state.who_raised}
- Available Actions: {', '.join(available_buttons)}

Based on this information:
1. Analyze the strength of your hand
2. Consider the pot odds and potential returns
3. Take into account previous actions
4. IMPORTANT: Only choose from the currently available actions listed above

Available actions:
{chr(10).join(action_options)}

Provide your response in JSON format:
{{
    "action": "{action_choices}",
    "amount": null/number (only for raise),
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation of your decision including hand strength analysis and strategic considerations"
}}

IMPORTANT: Your chosen action MUST be one of the available actions: {', '.join(available_buttons)}
Make sure to provide a detailed reasoning for your decision.
"""
        return prompt
    
    def _get_llm_response(self, prompt: str) -> str:
        """
        Call LLM API to get response
        Args:
            prompt: Input prompt for LLM
        Returns:
            str: LLM's response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional poker player. Analyze the game state and make strategic decisions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = f"Error getting LLM response: {str(e)}"
            logger.error(error_msg)
            return json.dumps({
                "action": "fold",
                "amount": None,
                "confidence": 0.0,
                "reasoning": f"Error occurred: {str(e)}, defaulting to fold"
            })
    
    def _parse_response(self, response: str, available_buttons: list) -> Dict[str, Any]:
        """
        Parse LLM's response and validate against available buttons
        Args:
            response: Response string from LLM
            available_buttons: List of currently available buttons
        Returns:
            Dict: Parsed action dictionary
        """
        # If no buttons are available, return wait action
        if not available_buttons:
            return {
                "action": "wait",
                "amount": None,
                "confidence": 0.0,
                "reasoning": "No action buttons are currently available, waiting for buttons to appear"
            }

        try:
            # Clean up the response string
            response = response.replace('\n', ' ').replace('\r', '')
            
            # Try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx + 1]
                # Clean up the JSON string
                json_str = json_str.replace('\t', ' ')
                # Remove multiple spaces
                while '  ' in json_str:
                    json_str = json_str.replace('  ', ' ')
                
                logger.debug(f"Cleaned JSON string:\n{json_str}")
                decision = json.loads(json_str)
                
                # Validate the action is available
                if decision["action"] not in available_buttons:
                    error_msg = f"LLM chose unavailable action: {decision['action']}"
                    logger.error(error_msg)
                    # Choose the most conservative available action
                    if "check" in available_buttons:
                        default_action = "check"
                    elif "fold" in available_buttons:
                        default_action = "fold"
                    else:
                        default_action = available_buttons[0]  # Safe now as we checked list is not empty
                    
                    return {
                        "action": default_action,
                        "amount": None,
                        "confidence": 0.0,
                        "reasoning": f"Original action {decision['action']} not available, defaulting to {default_action}"
                    }
                
                logger.info(f"Successfully parsed decision: {json.dumps(decision, indent=2)}")
                return decision
            else:
                error_msg = "No valid JSON found in response"
                logger.error(error_msg)
                # Safe to use available_buttons[0] as we checked list is not empty
                default_action = "fold" if "fold" in available_buttons else available_buttons[0]
                return {
                    "action": default_action,
                    "amount": None,
                    "confidence": 0.0,
                    "reasoning": f"Failed to find valid JSON in response, defaulting to {default_action}"
                }
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse response: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Problematic response:\n{response}")
            # Safe to use available_buttons[0] as we checked list is not empty
            default_action = "fold" if "fold" in available_buttons else available_buttons[0]
            return {
                "action": default_action,
                "amount": None,
                "confidence": 0.0,
                "reasoning": f"Failed to parse response, defaulting to {default_action}"
            }