import random
from typing import List, Optional, Union, Dict, Any

class RandomAgent:
    """A poker agent that makes random decisions."""
    
    def __init__(self):
        self.valid_actions = ['fold', 'call', 'raise']
    
    def calculate_action(self, 
                        community_cards: Optional[List[str]] = None,
                        player_hand: Optional[List[str]] = None,
                        pot: Optional[int] = None,
                        current_bet: Optional[int] = None) -> str:
        """
        Calculate a random action regardless of the game state.
        
        Args:
            community_cards: List of community cards (not used)
            player_hand: List of player's hole cards (not used)
            pot: Current pot size (not used)
            current_bet: Current bet to call (not used)
            
        Returns:
            str: A random action from ['fold', 'call', 'raise']
        """
        return random.choice(self.valid_actions)
    
    def reset(self):
        """Reset the agent's state if needed."""
        pass

# Function to be used directly without instantiating the class
def calculate_action(community_cards: Optional[List[str]] = None,
                    player_hand: Optional[List[str]] = None,
                    pot: Optional[int] = None,
                    current_bet: Optional[int] = None) -> str:
    """
    Standalone function to calculate a random action.
    
    Args:
        community_cards: List of community cards (not used)
        player_hand: List of player's hole cards (not used)
        pot: Current pot size (not used)
        current_bet: Current bet to call (not used)
        
    Returns:
        str: A random action from ['fold', 'call', 'raise']
    """
    return random.choice(['fold', 'call', 'raise']) 