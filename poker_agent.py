from typing import Dict, Any
import json
from openai import OpenAI
import os
from dotenv import load_dotenv

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
        
    def get_action(self, game_state) -> Dict[str, Any]:
        """
        Get action decision based on current game state
        Args:
            game_state: GameState instance containing current game state
        Returns:
            Dict: Dictionary containing action decision
        """
        # Build prompt
        prompt = self._build_prompt(game_state)
        
        # Get LLM response
        response = self._get_llm_response(prompt)
        
        # Parse LLM's decision
        return self._parse_response(response)
    
    def _build_prompt(self, game_state) -> str:
        """Build prompt for LLM"""
        return f"""
Given the current poker game state:
- Community Cards: {game_state.community_cards}
- Your Hole Cards: {game_state.hole_cards}
- Current Pot: {game_state.pot_size}
- Previous Actions: {game_state.who_raised}

What action should I take? Choose from:
1. Fold
2. Call
3. Raise (specify amount)

Provide your response in JSON format:
{{
    "action": "fold/call/raise",
    "amount": null/number (only for raise)
    "confidence": 0.0-1.0,
    "reasoning": "explanation"
}}
"""
    
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
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting LLM response: {str(e)}")
            return json.dumps({
                "action": "fold",
                "amount": None,
                "confidence": 0.0,
                "reasoning": "Error occurred, defaulting to fold"
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
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "action": "fold",
                "amount": None,
                "confidence": 0.0,
                "reasoning": "Failed to parse response, defaulting to fold"
            }