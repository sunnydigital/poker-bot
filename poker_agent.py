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
            game_state: Dictionary containing current game state
        Returns:
            Dict: Dictionary containing action decision
        """
        # Build prompt
        prompt = self._build_prompt(game_state)
        logger.debug(f"Generated prompt:\n{prompt}")
        
        # Get LLM response
        response = self._get_llm_response(prompt)
        logger.debug(f"Raw LLM response:\n{response}")
        
        # Parse LLM's decision
        decision = self._parse_response(response, game_state["legal_actions"])
        logger.debug(f"Parsed decision: {json.dumps(decision, indent=2)}")
        
        return decision
    
    def _build_prompt(self, game_state) -> str:
        """Build prompt for LLM"""
        # Convert available actions to a more readable format
        legal_actions = game_state["legal_actions"]
        action_descriptions = []
        if "fold" in legal_actions:
            action_descriptions.append("1. Fold - Give up the hand and lose any bets made")
        if "check" in legal_actions:
            action_descriptions.append("2. Check - Pass the action to the next player without betting")
        if "call" in legal_actions:
            action_descriptions.append(f"3. Call - Match the current bet of {game_state['current_bet']}")
        if "raise" in legal_actions:
            min_raise = game_state["current_bet"] * 2
            action_descriptions.append(f"4. Raise - Increase the bet (minimum raise: {min_raise})")
        
        action_choices = "/".join(legal_actions)
        
        prompt = f"""
You are a professional poker player. Analyze the current game state and make a strategic decision based on the AVAILABLE ACTIONS ONLY.

Current game state:
- Community Cards: {game_state["community_cards"]}
- Your Hole Cards: {game_state["hole_cards"]}
- Current Pot Size: {game_state["pot"]}
- Current Bet to Call: {game_state["current_bet"]}
- Your Stack: {game_state["stacks"][0]}
- Opponent Stack: {game_state["stacks"][1]}
- Stage: {game_state["stage"]}

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