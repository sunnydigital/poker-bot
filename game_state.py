import json
from typing import List, Dict, Any
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
        
        # Simulation attributes
        self.current_player: int = 0        # Current player's turn (0 or 1)
        self.player_stacks: List[float] = [1000, 1000]  # Starting stack for each player
        self.current_bet: float = 0         # Current bet amount
        self.min_raise: float = 2           # Minimum raise amount
        self.small_blind: float = 1         # Small blind amount
        self.big_blind: float = 2           # Big blind amount
        self.stage: str = "preflop"         # Current stage of the hand
        
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

    def reset(self):
        """Reset the game state to initial values"""
        self.community_cards = []
        self.hole_cards = []
        self.pot_size = 0
        self.who_raised = []
        # Add any other state variables that need to be reset

    def get_community_cards(self) -> List[str]:
        """Return the current community cards on the table"""
        return self.community_cards

    def get_player_hand(self, player_id: int) -> List[str]:
        """Return the hole cards for the specified player"""
        # For now, we'll assume player_id 0 refers to our player
        # and return the hole cards we know about
        if player_id == 0:
            return self.hole_cards
        else:
            # For other players, we don't know their cards
            return []
            
    def get_pot(self) -> int:
        """Return the current pot size"""
        return self.pot_size

    def get_current_bet(self) -> float:
        """Return the current bet amount on the table"""
        # If there are any raises, return the last raise amount
        if self.who_raised:
            return self.who_raised[-1].get('amount', 0)
        return 0

    def apply_action(self, player: int, action: Dict[str, Any]) -> bool:
        """
        Apply a player's action to the game state
        Args:
            player: Player index (0 or 1)
            action: Action dictionary with keys: action, amount
        Returns:
            bool: True if the hand is complete, False otherwise
        """
        action_type = action["action"].lower()
        amount = action.get("amount", 0)
        
        # Record the action
        self.last_action = action_type
        self.last_action_amount = amount
        
        # Update game state based on action
        if action_type == "fold":
            self.is_game_over = True
            # Other player wins the pot
            winner = 1 if player == 0 else 0
            self.player_stacks[winner] += self.pot_size
            self.result = "lose" if player == 0 else "win"
            self.stack_change = -self.current_bet if player == 0 else self.pot_size
            return True
            
        elif action_type in ["call", "check"]:
            if action_type == "call":
                call_amount = self.current_bet - self.who_raised[-1].get('amount', 0) if self.who_raised else self.current_bet
                self.pot_size += call_amount
                self.player_stacks[player] -= call_amount
            
            # If both players have acted, move to next stage
            if len(self.who_raised) >= 2:
                return self._advance_stage()
                
        elif action_type == "raise":
            raise_amount = max(amount, self.min_raise)
            self.current_bet = raise_amount
            self.pot_size += raise_amount
            self.player_stacks[player] -= raise_amount
            self.who_raised.append({
                "player": player,
                "amount": raise_amount
            })
            
        # Switch to next player
        self.current_player = 1 if player == 0 else 0
        return False
        
    def _advance_stage(self) -> bool:
        """
        Advance to the next stage of the hand
        Returns:
            bool: True if the hand is complete, False otherwise
        """
        if self.stage == "preflop":
            self.stage = "flop"
            # Deal flop
            self.community_cards = ["7-H", "8-D", "9-C"]  # Simplified for simulation
        elif self.stage == "flop":
            self.stage = "turn"
            self.community_cards.append("10-S")  # Add turn card
        elif self.stage == "turn":
            self.stage = "river"
            self.community_cards.append("J-H")  # Add river card
        elif self.stage == "river":
            self.is_game_over = True
            # Determine winner (simplified)
            self._evaluate_winner()
            return True
            
        # Reset betting round
        self.current_bet = 0
        self.who_raised = []
        self.current_player = 0  # Start with first player
        return False
        
    def _evaluate_winner(self):
        """Simplified winner evaluation for simulation"""
        # For simulation, randomly determine winner
        import random
        if random.random() < 0.5:
            self.result = "win"
            self.stack_change = self.pot_size / 2
            self.player_stacks[0] += self.pot_size
        else:
            self.result = "lose"
            self.stack_change = -self.pot_size / 2
            self.player_stacks[1] += self.pot_size
            
    def get_legal_actions(self, player: int) -> List[str]:
        """
        Get list of legal actions for the current player
        Args:
            player: Player index (0 or 1)
        Returns:
            List[str]: List of legal action types
        """
        legal_actions = ["fold"]
        
        # Can only check if no bet to call
        if not self.current_bet or (self.who_raised and self.who_raised[-1].get('amount', 0) == self.current_bet):
            legal_actions.append("check")
        else:
            legal_actions.append("call")
            
        # Can always raise if have enough chips
        if self.player_stacks[player] >= self.current_bet + self.min_raise:
            legal_actions.append("raise")
            
        return legal_actions