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
    def __init__(self, model_name: str = "gpt-4o-mini"):
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
        # Log current game state
        logger.info("Current Game State:")
        logger.info(f"Community Cards: {game_state.community_cards}")
        logger.info(f"Hole Cards: {game_state.hole_cards}")
        logger.info(f"Pot Size: {game_state.pot_size}")
        logger.info(f"Previous Actions: {game_state.who_raised}")
        
        # Build prompt
        prompt = self._build_prompt(game_state)
        logger.debug(f"Generated prompt:\n{prompt}")
        
        # Get LLM response
        response = self._get_llm_response(prompt)
        logger.info(f"Raw LLM response:\n{response}")
        
        # Parse LLM's decision
        decision = self._parse_response(response)
        logger.info(f"Parsed decision: {json.dumps(decision, indent=2)}")
        
        return decision
    
    def _build_prompt(self, game_state) -> str:
        """Build prompt for LLM"""
        prompt = f"""
You are a professional poker player. Analyze the current game state and make a strategic decision.

Current game state:
- Community Cards: {game_state.community_cards}
- Your Hole Cards: {game_state.hole_cards}
- Current Pot Size: {game_state.pot_size}
- Previous Actions: {game_state.who_raised}

Based on this information:
1. Analyze the strength of your hand
2. Consider the pot odds and potential returns
3. Take into account previous actions

Choose your action from:
1. Fold - if the hand is weak and not worth playing
2. Call - if the hand has potential but not strong enough to raise
3. Raise (specify amount) - if you have a strong hand or good bluffing opportunity

Provide your response in JSON format:
{{
    "action": "fold/call/raise",
    "amount": null/number (only for raise),
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation of your decision including hand strength analysis and strategic considerations"
}}

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
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM's response
        Args:
            response: Response string from LLM
        Returns:
            Dict: Parsed action dictionary
        """
        try:
            # Try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx + 1]
                decision = json.loads(json_str)
                return decision
            else:
                error_msg = "No valid JSON found in response"
                logger.error(error_msg)
                return {
                    "action": "fold",
                    "amount": None,
                    "confidence": 0.0,
                    "reasoning": "Failed to find valid JSON in response, defaulting to fold"
                }
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse response: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Problematic response:\n{response}")
            return {
                "action": "fold",
                "amount": None,
                "confidence": 0.0,
                "reasoning": "Failed to parse response, defaulting to fold"
            }