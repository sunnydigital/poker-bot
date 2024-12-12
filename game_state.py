import json
from typing import List, Dict
import base64
from io import BytesIO
from PIL import Image
import requests
from utils.llm_api import LLMAnalyzer
import copy

class GameState:
    def __init__(self, api_key: str = None):
        self.community_cards: List[str] = []  # Community cards on the table ["A-S", "2-H", "3-C"]
        self.hole_cards: List[str] = []      # Player's hole cards ["5-D", "6-S"]
        self.pot_size: int = 0              # Current pot size
        self.is_game_over: bool = False     # Whether the game is over
        self.who_raised: List[Dict] = []    # Player raise information
        self.llm_analyzer = LLMAnalyzer(api_key)
        
        # Game tracking attributes
        self.hand_id: str = ""              # Unique identifier for each hand
        self.last_action: str = ""          # Last action taken
        self.last_action_amount: float = 0  # Amount of last action
        self.result: str = ""               # Result of the hand (win/lose/draw)
        self.stack_change: float = 0        # Change in stack size
        
    def copy(self) -> 'GameState':
        """Create a deep copy of the game state"""
        new_state = GameState()
        new_state.community_cards = self.community_cards.copy()
        new_state.hole_cards = self.hole_cards.copy()
        new_state.pot_size = self.pot_size
        new_state.is_game_over = self.is_game_over
        new_state.who_raised = copy.deepcopy(self.who_raised)
        new_state.hand_id = self.hand_id
        new_state.last_action = self.last_action
        new_state.last_action_amount = self.last_action_amount
        new_state.result = self.result
        new_state.stack_change = self.stack_change
        return new_state
        
    def _image_to_base64(self, image) -> str:
        """
        Convert numpy image to base64 string
        Args:
            image: numpy array format image
        Returns:
            str: base64 encoded image string
        """
        img = Image.fromarray(image)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def update_from_screen(self, screen_image):
        """
        Update game state using LLM to analyze screen image
        Args:
            screen_image: numpy.ndarray format screenshot
        """
        # Convert image to base64
        image_base64 = self._image_to_base64(screen_image)
        
        # Call LLM API for analysis
        response = self.llm_analyzer.analyze_poker_image(image_base64)
        
        # Parse returned JSON
        state = json.loads(response['boardState'])
        
        # Update game state
        self.community_cards = state['communityCards']
        self.hole_cards = state['holeCards']
        self.pot_size = state['currentPotValue']
        self.is_game_over = state['isGameOver']
        self.who_raised = state['whoRaised']
        
        # Generate or update hand ID based on hole cards and community cards
        self.hand_id = f"{'-'.join(sorted(self.hole_cards))}_{'-'.join(sorted(self.community_cards))}"
        
        # Update result and stack change if game is over
        if self.is_game_over:
            self.result = state.get('result', '')
            self.stack_change = state.get('stackChange', 0)
        
        # Print current state (for debugging)
        print("Current Game State:")
        print(f"Community Cards: {self.community_cards}")
        print(f"Hole Cards: {self.hole_cards}")
        print(f"Pot Size: {self.pot_size}")
        print(f"Game Over: {self.is_game_over}")
        print(f"Who Raised: {self.who_raised}")