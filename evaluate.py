import random
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime
from pathlib import Path
from poker_agent import PokerAgent
from gto_agent import calculate_action as rule_based_calculate_action
from random_agent import calculate_action as random_calculate_action

# Constants
NUM_CARDS = 52
STARTING_STACK = 1000
SMALL_BLIND = 10
BIG_BLIND = 20
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
SUITS = ['H', 'D', 'C', 'S']

# Create results directory
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

class HandResult:
    def __init__(self, hand_num: int):
        self.hand_num = hand_num
        self.actions = []  # List of (player, stage, action) tuples
        self.community_cards = []
        self.player_hands = []
        self.winner = ""
        self.pot = 0
        self.final_stacks = []
        self.stages_reached = []

class PokerEnvEvaluator:
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset the game environment"""
        self.deck = self._create_deck()
        random.shuffle(self.deck)
        
        # Deal hole cards
        self.player_hands = [
            [self.deck.pop(), self.deck.pop()],
            [self.deck.pop(), self.deck.pop()],
            [self.deck.pop(), self.deck.pop()]
        ]
        
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.stage = "preflop"
        self.current_player = 0
        self.player_stacks = [STARTING_STACK, STARTING_STACK, STARTING_STACK]
        self.folded_players = [False, False, False]  # Track who has folded
        self.last_action = None
        self.last_raise = 0
        self.num_actions = 0
        
        # Post blinds
        self.player_stacks[0] -= SMALL_BLIND
        self.player_stacks[1] -= BIG_BLIND
        self.pot = SMALL_BLIND + BIG_BLIND
        self.current_bet = BIG_BLIND
        
        return self._get_state()
    
    def _create_deck(self) -> List[str]:
        """Create a standard deck of 52 cards"""
        return [f"{rank}-{suit}" for rank in RANKS for suit in SUITS]
    
    def _get_state(self) -> Dict:
        """Get current game state"""
        return {
            "community_cards": self.community_cards.copy(),
            "hole_cards": self.player_hands[self.current_player].copy(),
            "pot": self.pot,
            "current_bet": self.current_bet,
            "stage": self.stage,
            "stacks": self.player_stacks.copy(),
            "legal_actions": self.get_legal_actions(),
            "last_action": self.last_action,
            "last_raise": self.last_raise,
            "folded": self.folded_players[self.current_player]
        }
    
    def get_legal_actions(self) -> List[str]:
        """Get list of legal actions for current player"""
        legal_actions = ["fold"]
        
        # Can only check if no bet to call or already matched the bet
        if self.current_bet == 0 or (self.last_action == "raise" and self.num_actions >= 2):
            legal_actions.append("check")
        else:
            legal_actions.append("call")
        
        # Can raise if have enough chips and not already raised twice
        min_raise = max(self.current_bet * 2, BIG_BLIND)
        if self.player_stacks[self.current_player] >= min_raise and self.num_actions < 4:
            legal_actions.append("raise")
            
        return legal_actions
    
    def _deal_community_cards(self):
        """Deal community cards based on current stage"""
        if self.stage == "preflop":
            self.community_cards.extend([self.deck.pop() for _ in range(3)])
            self.stage = "flop"
        elif self.stage == "flop":
            self.community_cards.append(self.deck.pop())
            self.stage = "turn"
        elif self.stage == "turn":
            self.community_cards.append(self.deck.pop())
            self.stage = "river"
    
    def _evaluate_hand(self, hole_cards: List[str], community_cards: List[str]) -> int:
        """Evaluate hand strength"""
        all_cards = hole_cards + community_cards
        all_card_values = [(RANKS.index(card.split('-')[0]), card.split('-')[1]) for card in all_cards]
        
        # Sort by rank
        all_card_values.sort(key=lambda x: x[0], reverse=True)
        
        # Get ranks and suits
        ranks = [v[0] for v in all_card_values]
        suits = [v[1] for v in all_card_values]
        
        # Check for flush
        suit_counts = {}
        for suit in suits:
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
        
        flush_suit = None
        for suit, count in suit_counts.items():
            if count >= 5:
                flush_suit = suit
                break
        
        # Check for straight
        straight_high = None
        for i in range(len(ranks) - 4):
            if ranks[i] - ranks[i+4] == 4:
                straight_high = ranks[i]
                break
        
        # Count rank frequencies
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        # Find highest frequency
        max_freq = max(rank_counts.values())
        
        # Calculate hand value
        if flush_suit and straight_high:
            return 8000 + straight_high  # Straight flush
        elif max_freq == 4:
            return 7000 + max(r for r, f in rank_counts.items() if f == 4)  # Four of a kind
        elif max_freq == 3 and 2 in rank_counts.values():
            return 6000 + max(r for r, f in rank_counts.items() if f == 3)  # Full house
        elif flush_suit:
            return 5000 + max(ranks)  # Flush
        elif straight_high:
            return 4000 + straight_high  # Straight
        elif max_freq == 3:
            return 3000 + max(r for r, f in rank_counts.items() if f == 3)  # Three of a kind
        elif list(rank_counts.values()).count(2) == 2:
            pairs = sorted([r for r, f in rank_counts.items() if f == 2], reverse=True)
            return 2000 + pairs[0] * 13 + pairs[1]  # Two pair
        elif max_freq == 2:
            return 1000 + max(r for r, f in rank_counts.items() if f == 2)  # One pair
        else:
            return max(ranks)  # High card
    
    def step(self, action: str) -> Tuple[Dict, float, bool, Dict]:
        """Execute one step in the game"""
        self.last_action = action
        self.num_actions += 1
        
        if action == "fold":
            # Mark player as folded
            self.folded_players[self.current_player] = True
            
            # Count folded players
            num_folded = sum(self.folded_players)
            
            # If all but one player has folded, end the hand
            if num_folded >= 2:
                # Find the player who hasn't folded
                winner = self.folded_players.index(False)
                self.player_stacks[winner] += self.pot
                reward = self.pot if winner == 0 else -self.pot
                return self._get_state(), reward, True, {}
            
            # Move to next active player
            self.current_player = self._next_active_player()
            return self._get_state(), 0, False, {}
            
        elif action == "call":
            # Match the current bet
            call_amount = self.current_bet - self.last_raise
            self.player_stacks[self.current_player] -= call_amount
            self.pot += call_amount
            
            # If all non-folded players have acted, check if we should move to next stage
            active_players = len([p for p in self.folded_players if not p])
            if self.num_actions >= active_players:
                if self.stage == "river":
                    # Evaluate hands for non-folded players
                    hand_values = [
                        self._evaluate_hand(self.player_hands[i], self.community_cards)
                        if not self.folded_players[i] else -1
                        for i in range(3)
                    ]
                    winner = hand_values.index(max(hand_values))
                    self.player_stacks[winner] += self.pot
                    reward = self.pot if winner == 0 else -self.pot
                    return self._get_state(), reward, True, {}
                else:
                    # Move to next stage
                    self._deal_community_cards()
                    self.current_bet = 0
                    self.last_raise = 0
                    self.num_actions = 0
                    self.current_player = self._first_active_player()
                    return self._get_state(), 0, False, {}
            
            # Move to next active player
            self.current_player = self._next_active_player()
            return self._get_state(), 0, False, {}
            
        elif action == "check":
            # If all non-folded players have acted, check if we should move to next stage
            active_players = len([p for p in self.folded_players if not p])
            if self.num_actions >= active_players:
                if self.stage == "river":
                    # Evaluate hands for non-folded players
                    hand_values = [
                        self._evaluate_hand(self.player_hands[i], self.community_cards)
                        if not self.folded_players[i] else -1
                        for i in range(3)
                    ]
                    winner = hand_values.index(max(hand_values))
                    self.player_stacks[winner] += self.pot
                    reward = self.pot if winner == 0 else -self.pot
                    return self._get_state(), reward, True, {}
                else:
                    # Move to next stage
                    self._deal_community_cards()
                    self.current_bet = 0
                    self.last_raise = 0
                    self.num_actions = 0
                    self.current_player = self._first_active_player()
                    return self._get_state(), 0, False, {}
            
            # Move to next active player
            self.current_player = self._next_active_player()
            return self._get_state(), 0, False, {}
            
        elif action == "raise":
            # Make a raise
            raise_amount = self.current_bet * 2
            self.player_stacks[self.current_player] -= raise_amount
            self.pot += raise_amount
            self.last_raise = raise_amount
            self.current_bet = raise_amount
            
            # Move to next active player
            self.current_player = self._next_active_player()
            return self._get_state(), 0, False, {}
            
        return self._get_state(), 0, False, {}

    def _next_active_player(self) -> int:
        """Find the next player who hasn't folded"""
        current = self.current_player
        while True:
            current = (current + 1) % 3
            if not self.folded_players[current]:
                return current
            if current == self.current_player:  # We've gone full circle
                return current

    def _first_active_player(self) -> int:
        """Find the first player who hasn't folded"""
        for i in range(3):
            if not self.folded_players[i]:
                return i
        return 0  # Shouldn't happen if we handle folds correctly

def simulate_hand(env: PokerEnvEvaluator, gto_agent, random_agent, llm_agent, hand_num: int) -> HandResult:
    state = env.reset()
    done = False
    result = HandResult(hand_num)
    result.player_hands = env.player_hands.copy()
    
    print(f"\nHand {hand_num + 1}:")
    
    while not done:
        if env.current_player == 0:  # GTO agent's turn
            community_cards = env.community_cards
            hole_cards = env.player_hands[env.current_player]
            action = gto_agent(community_cards, hole_cards, env.pot, [env.current_bet])
            print(f"GTO Agent action: {action}")
        elif env.current_player == 1:  # Random agent's turn
            action = random_agent(None, None, None, None)
            print(f"Random Agent action: {action}")
        else:  # LLM agent's turn
            action = llm_agent.get_action(state)["action"]
            if action not in state["legal_actions"]:
                action = state["legal_actions"][0]
            print(f"LLM Agent action: {action}")
            
        # Record action
        result.actions.append((
            "GTO" if env.current_player == 0 else "Random" if env.current_player == 1 else "LLM",
            env.stage,
            action
        ))
        
        # Take action
        next_state, reward, done, _ = env.step(action)
        
        # Record stage if it changed
        if next_state["stage"] != state["stage"] and next_state["stage"] not in result.stages_reached:
            result.stages_reached.append(next_state["stage"])
        
        state = next_state
        
    # Record final state
    result.community_cards = env.community_cards.copy()
    result.pot = env.pot
    result.final_stacks = env.player_stacks.copy()
    
    # Determine winner
    stack_changes = [stack - STARTING_STACK for stack in result.final_stacks]
    winner_idx = stack_changes.index(max(stack_changes))
    result.winner = "GTO" if winner_idx == 0 else "Random" if winner_idx == 1 else "LLM"
    
    print(f"Hand {hand_num + 1} result: {result.winner} wins")
    return result

def simulate_game(env: PokerEnvEvaluator, gto_agent, random_agent, llm_agent, num_hands: int):
    # Initialize results tracking
    results = []
    win_history = {'GTO': [], 'Random': [], 'LLM': []}
    wins = {'GTO': 0, 'Random': 0, 'LLM': 0}
    
    for hand_num in range(num_hands):
        result = simulate_hand(env, gto_agent, random_agent, llm_agent, hand_num)
        results.append(result)
        
        # Update win counts
        wins[result.winner] += 1
        win_history['GTO'].append(wins['GTO'])
        win_history['Random'].append(wins['Random'])
        win_history['LLM'].append(wins['LLM'])
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed hand results
    hand_data = []
    for result in results:
        hand_data.append({
            'hand_num': result.hand_num,
            'winner': result.winner,
            'pot': result.pot,
            'community_cards': result.community_cards,
            'player_hands': result.player_hands,
            'final_stacks': result.final_stacks,
            'stages_reached': result.stages_reached,
            'actions': result.actions
        })
    
    with open(RESULTS_DIR / f'hand_results_{timestamp}.json', 'w') as f:
        json.dump(hand_data, f, indent=2)
    
    # Save win history
    win_df = pd.DataFrame(win_history)
    win_df.to_csv(RESULTS_DIR / f'win_history_{timestamp}.csv', index=False)
    
    # Save action statistics
    action_data = []
    for result in results:
        for player, stage, action in result.actions:
            action_data.append({
                'hand_num': result.hand_num,
                'player': player,
                'stage': stage,
                'action': action,
                'winner': result.winner
            })
    
    action_df = pd.DataFrame(action_data)
    action_df.to_csv(RESULTS_DIR / f'action_stats_{timestamp}.csv', index=False)
    
    # Print summary
    print(f"\nResults after {num_hands} hands:")
    for player, win_count in wins.items():
        print(f"{player} Agent Wins: {win_count} ({win_count/num_hands*100:.1f}%)")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    for player, history in win_history.items():
        plt.plot(range(1, num_hands + 1), history, label=f'{player} Wins')
    plt.xlabel('Number of Hands')
    plt.ylabel('Cumulative Wins')
    plt.title('Cumulative Wins Over Hands')
    plt.legend()
    plt.grid(True)
    plt.savefig(RESULTS_DIR / f'win_plot_{timestamp}.png')
    plt.show()
    
    return {
        'timestamp': timestamp,
        'num_hands': num_hands,
        'wins': wins,
        'win_history': win_history,
        'results': results
    }

if __name__ == "__main__":
    env = PokerEnvEvaluator()
    llm_agent = PokerAgent(model_name="gpt-4o")
    
    num_hands = 100
    results = simulate_game(env, gto_calculate_action, random_calculate_action, llm_agent, num_hands) 