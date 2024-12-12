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
    def __init__(self, model_name: str = "gpt-4"):
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
            game_state: Dictionary containing current game state or GameState object
        Returns:
            Dict: Dictionary containing action decision
        """
        # Build prompt
        prompt = self._build_prompt(game_state)
        logger.debug(f"Generated prompt:\n{prompt}")
        
        # Get LLM response
        response = self._get_llm_response(prompt)
        logger.debug(f"Raw LLM response:\n{response}")
        
        # Get legal actions based on game state type
        legal_actions = self._get_legal_actions(game_state)
        
        # Parse LLM's decision
        decision = self._parse_response(response, legal_actions)
        logger.debug(f"Parsed decision: {json.dumps(decision, indent=2)}")
        
        # Print decision details
        print("\nLLM Decision:")
        print(f"Action: {decision['action']}")
        if decision.get('amount'):
            print(f"Amount: {decision['amount']}")
        print(f"Confidence: {decision.get('confidence', 'N/A')}")
        print(f"Reasoning: {decision.get('reasoning', 'No reasoning provided')}\n")
        
        return decision

    def _get_legal_actions(self, game_state) -> list:
        """Get legal actions based on game state type"""
        if isinstance(game_state, dict):
            return game_state.get("legal_actions", ["fold", "call", "raise"])
        else:
            # For GameState object, we'll use default poker actions
            return ["fold", "call", "raise"]

    def _get_game_state_info(self, game_state) -> Dict[str, Any]:
        """Extract game state information regardless of input type"""
        if isinstance(game_state, dict):
            return {
                "community_cards": game_state.get("community_cards", []),
                "hole_cards": game_state.get("hole_cards", []),
                "pot": game_state.get("pot", 0),
                "current_bet": game_state.get("current_bet", 0),
                "stacks": game_state.get("stacks", [1000, 1000]),
                "stage": game_state.get("stage", "preflop")
            }
        else:
            # Handle GameState object - using correct attribute names
            try:
                # Get attributes safely with getattr
                community_cards = getattr(game_state, "community_cards", [])
                hole_cards = getattr(game_state, "hole_cards", [])
                pot = getattr(game_state, "pot_size", 0)
                current_bet = getattr(game_state, "current_bet", 0)
                stacks = getattr(game_state, "player_stacks", [1000, 1000])
                stage = getattr(game_state, "stage", "preflop")
                
                return {
                    "community_cards": community_cards,
                    "hole_cards": hole_cards,
                    "pot": pot,
                    "current_bet": current_bet,
                    "stacks": stacks,
                    "stage": stage
                }
            except Exception as e:
                logger.error(f"Error accessing GameState attributes: {e}")
                # Provide safe default values
                return {
                    "community_cards": [],
                    "hole_cards": [],
                    "pot": 0,
                    "current_bet": 0,
                    "stacks": [1000, 1000],
                    "stage": "preflop"
                }

    def _build_prompt(self, game_state) -> str:
        """Build prompt for LLM"""
        # Get game state info and legal actions
        state_info = self._get_game_state_info(game_state)
        legal_actions = self._get_legal_actions(game_state)
        
        # Convert available actions to a more readable format
        action_descriptions = []
        if "fold" in legal_actions:
            action_descriptions.append("1. Fold - Give up the hand and lose any bets made")
        if "check" in legal_actions:
            action_descriptions.append("2. Check - Pass the action to the next player without betting")
        if "call" in legal_actions:
            action_descriptions.append(f"3. Call - Match the current bet of {state_info['current_bet']}")
        if "raise" in legal_actions:
            min_raise = state_info['current_bet'] * 2
            action_descriptions.append(f"4. Raise - Increase the bet (minimum raise: {min_raise})")
        
        action_choices = "/".join(legal_actions)
        
        prompt = f"""
You are a professional poker player. Analyze the current game state and make a strategic decision based on the AVAILABLE ACTIONS ONLY.

Current game state:
- Community Cards: {state_info['community_cards']}
- Your Hole Cards: {state_info['hole_cards']}
- Current Pot Size: {state_info['pot']}
- Current Bet to Call: {state_info['current_bet']}
- Your Stack: {state_info['stacks'][0]}
- Opponent Stack: {state_info['stacks'][1]}
- Stage: {state_info['stage']}

AVAILABLE ACTIONS:
{chr(10).join(action_descriptions)}

Based on this information:
1. Analyze the strength of your hand
2. Consider the pot odds and potential returns
3. Take into account stack sizes and betting patterns
4. IMPORTANT: You MUST choose one of these actions: {action_choices}

Provide your response in JSON format:
{{
    "action": "{action_choices}",
    "amount": null/number (only for raise),
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation of your decision including hand strength analysis and strategic considerations"
}}

IMPORTANT: Your chosen action MUST be one of: {action_choices}
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
    
    def _parse_response(self, response: str, legal_actions: list) -> Dict[str, Any]:
        """
        Parse LLM's response and validate against legal actions
        Args:
            response: Response string from LLM
            legal_actions: List of currently legal actions
        Returns:
            Dict: Parsed action dictionary
        """
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
                
                # Validate the action is legal
                if decision["action"] not in legal_actions:
                    error_msg = f"LLM chose illegal action: {decision['action']}"
                    logger.error(error_msg)
                    # Choose the most conservative legal action
                    if "check" in legal_actions:
                        default_action = "check"
                    elif "fold" in legal_actions:
                        default_action = "fold"
                    else:
                        default_action = legal_actions[0]
                    
                    return {
                        "action": default_action,
                        "amount": None,
                        "confidence": 0.0,
                        "reasoning": f"Original action {decision['action']} not legal, defaulting to {default_action}"
                    }
                
                # Ensure all required fields are present
                if 'confidence' not in decision:
                    decision['confidence'] = 0.8  # Default confidence
                if 'reasoning' not in decision:
                    decision['reasoning'] = "No explicit reasoning provided"
                if 'amount' not in decision:
                    decision['amount'] = None
                
                logger.debug(f"Successfully parsed decision: {json.dumps(decision, indent=2)}")
                return decision
            else:
                error_msg = "No valid JSON found in response"
                logger.error(error_msg)
                default_action = "fold" if "fold" in legal_actions else legal_actions[0]
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
            default_action = "fold" if "fold" in legal_actions else legal_actions[0]
            return {
                "action": default_action,
                "amount": None,
                "confidence": 0.0,
                "reasoning": f"Failed to parse response, defaulting to {default_action}"
            }