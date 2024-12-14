from typing import List
import itertools
import random
from collections import Counter

def calculate_action(community_cards: List[str], hole_cards: List[str], currentPotValue: int, raiseAmounts: List[int]) -> str:
    """
    Calculate the optimal action for the agent based on the given game state.
    """
    # Combine community cards and hole cards
    all_cards = community_cards + hole_cards

    # Determine the strength of the hand
    hand_strength = evaluate_hand_strength(all_cards, hole_cards)
    raisedBy = max(raiseAmounts) if raiseAmounts else 0
    betsBeenMade = raisedBy != 0

    # Calculate the pot odds
    call_amount = raisedBy
    pot_odds = call_amount / (currentPotValue + call_amount) if call_amount > 0 else 0

    # Estimate the probability of winning based on Monte Carlo simulation
    win_probability = simulate_win_probability(community_cards, hole_cards)

    # Calculate EV for Call and Raise
    ev_call = win_probability * (currentPotValue + call_amount) - (1 - win_probability) * call_amount
    ev_raise = win_probability * (currentPotValue + 2 * call_amount) - (1 - win_probability) * 2 * call_amount

    # Make a decision based on EV and pot odds
    if not community_cards:
        # Pre-flop strategy
        if hand_strength < 1.5:
            return "fold"
        if betsBeenMade:
            if ev_call > 0:
                if ev_raise > ev_call * 1.2:  # Strong hand
                    return "raise"
                return "call"
            else:
                return "fold"
        return "raise" if hand_strength > 2 else "call"
    else:
        # Post-flop strategy
        if betsBeenMade:
            if ev_raise > ev_call * 1.5:  # Very strong hand
                return "raise"
            elif ev_call > 0:
                return "call"
            else:
                return "fold"
        else:
            if win_probability > 0.7:  # Strong hand
                return "raise"
            return "check"

def evaluate_hand_strength(cards: List[str], hole_cards: List[str]) -> float:
    """
    Evaluate the strength of the hand based on common poker rules.
    """
    # Convert cards to a format suitable for evaluation
    formatted_cards = [card.split('-') for card in cards]
    ranks = [rank for rank, _ in formatted_cards]  # Keep '10' as it is
    suits = [suit for _, suit in formatted_cards]

    # Update rank_values to include '10' as a key
    rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

    # Convert ranks to numeric values using rank_values
    try:
        numeric_ranks = [rank_values[rank] for rank in ranks]
    except KeyError as e:
        raise ValueError(f"Invalid rank {e.args[0]} found in cards. Valid ranks are: {list(rank_values.keys())}")

    # Sort the numeric ranks in ascending order
    sorted_ranks = sorted(numeric_ranks)

    # Calculate the value of the hand based on the current heuristics
    hand_value = 0
    if is_flush(suits) and is_straight(sorted_ranks):
        hand_value = 9
    elif is_four_of_a_kind(sorted_ranks):
        hand_value = 8
    elif is_full_house(sorted_ranks):
        hand_value = 7
    elif is_flush(suits):
        hand_value = 6
    elif is_straight(sorted_ranks):
        hand_value = 5
    elif is_three_of_a_kind(sorted_ranks):
        hand_value = 4
    elif is_two_pair(sorted_ranks):
        hand_value = 3
    elif is_one_pair(sorted_ranks):
        hand_value = 2
    else:
        hand_value = 1

    # Evaluate hole cards
    hole_card_ranks = [rank_values[card.split('-')[0]] for card in hole_cards]
    high_card = max(hole_card_ranks)

    # Add a slight bonus for high cards
    hand_value += high_card / 14.0

    return hand_value

def simulate_win_probability(community_cards: List[str], hole_cards: List[str], num_simulations: int = 1000) -> float:
    """
    Simulate the probability of winning using a Monte Carlo approach.
    """
    deck = [f"{rank}-{suit}" for rank in ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'] for suit in 'SHDC']
    used_cards = set(community_cards + hole_cards)
    remaining_deck = [card for card in deck if card not in used_cards]
    wins = 0

    for _ in range(num_simulations):
        random.shuffle(remaining_deck)
        opponent_hole_cards = remaining_deck[:2]
        remaining_community_cards = remaining_deck[2:5-len(community_cards)]
        final_community_cards = community_cards + remaining_community_cards

        player_score = evaluate_hand_strength(final_community_cards + hole_cards, hole_cards)
        opponent_score = evaluate_hand_strength(final_community_cards + opponent_hole_cards, opponent_hole_cards)

        if player_score > opponent_score:
            wins += 1

    return wins / num_simulations

def is_flush(suits: List[str]) -> bool:
    return any(len(set(hand)) == 1 for hand in itertools.combinations(suits, 5))

def is_straight(sorted_ranks: List[int]) -> bool:
    return any(hand == tuple(range(min(hand), max(hand) + 1)) 
              for hand in itertools.combinations(sorted_ranks, 5))

def is_four_of_a_kind(ranks: List[int]) -> bool:
    return max(Counter(ranks).values()) == 4

def is_full_house(ranks: List[int]) -> bool:
    rank_counts = Counter(ranks).values()
    return 3 in rank_counts and 2 in rank_counts

def is_three_of_a_kind(ranks: List[int]) -> bool:
    return max(Counter(ranks).values()) == 3

def is_two_pair(ranks: List[int]) -> bool:
    rank_counts = Counter(ranks).values()
    return list(rank_counts).count(2) == 2

def is_one_pair(ranks: List[int]) -> bool:
    rank_counts = Counter(ranks).values()
    return 2 in rank_counts
