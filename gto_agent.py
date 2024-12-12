from typing import List
import itertools

def calculate_action(community_cards: List[str], hole_cards: List[str], currentPotValue: int, raiseAmounts: List[int]) -> str:
    # Combine community cards and hole cards
    all_cards = community_cards + hole_cards

    # Determine the strength of the hand
    hand_strength = evaluate_hand_strength(all_cards, hole_cards)
    raisedBy = max(raiseAmounts) if raiseAmounts else 0
    betsBeenMade = raisedBy != 0

    # Calculate the pot odds
    call_amount = raisedBy
    pot_odds = call_amount / (currentPotValue + call_amount) if call_amount > 0 else 0

    # Estimate the probability of winning based on hand strength
    win_probability = hand_strength / 9

    # Make a decision based on hand strength and pot odds
    if not community_cards:
        # Pre-flop strategy
        if hand_strength < 1.45:
            return "fold"
        if betsBeenMade:
            if win_probability > pot_odds / 2:
                if win_probability > 0.7:  # Strong hand
                    return "raise"
                return "call"
            else:
                return "fold"
        return "raise" if win_probability > 0.6 else "call"
    else:
        # Post-flop strategy
        if betsBeenMade:
            if win_probability > pot_odds * 1.5:  # Very strong hand
                return "raise"
            elif win_probability > pot_odds:
                return "call"
            else:
                return "fold"
        else:
            if win_probability > 0.7:  # Strong hand
                return "raise"
            return "check"

def evaluate_hand_strength(cards: List[str], hole_cards: List[str]) -> float:
    # Convert cards to a format suitable for evaluation
    formatted_cards = [card.split('-') for card in cards]
    ranks = [rank for rank, _ in formatted_cards]
    suits = [suit for _, suit in formatted_cards]
    rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    numeric_ranks = [rank_values[str(rank)] for rank in ranks]

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

    hole_card_ranks = [rank_values[card.split('-')[0]] for card in hole_cards]
    hole_card_suits = [card.split('-')[1] for card in hole_cards]

    high_card = max(hole_card_ranks)
    low_card = min(hole_card_ranks)

    # Calculate the value of the hole cards (0-1) based on their ranks and suits
    hole_card_value = 0
    if hole_card_suits[0] == hole_card_suits[1]:  # Suited cards
        if high_card == 14:  # Ace
            hole_card_value = 0.9
        elif high_card >= 12:  # Queen or higher
            hole_card_value = 0.8 - (high_card - 12) * 0.05
        elif high_card >= 8:  # Ten or higher
            hole_card_value = 0.6 - (high_card - 8) * 0.05
        else:
            hole_card_value = 0.4 - (high_card - 2) * 0.03
    elif abs(high_card - low_card) <= 4:  # Connected cards
        if high_card == 14:  # Ace
            hole_card_value = 0.8
        elif high_card >= 12:  # Queen or higher
            hole_card_value = 0.7 - (high_card - 12) * 0.05
        elif high_card >= 8:  # Ten or higher
            hole_card_value = 0.5 - (high_card - 8) * 0.05
        else:
            hole_card_value = 0.3 - (high_card - 2) * 0.03
    else:
        if high_card == 14:  # Ace
            hole_card_value = 0.7
        elif high_card >= 12:  # Queen or higher
            hole_card_value = 0.6 - (high_card - 12) * 0.1
        elif high_card >= 8:  # Ten or higher
            hole_card_value = 0.4 - (high_card - 8) * 0.05
        else:
            hole_card_value = 0.2 - (high_card - 2) * 0.02

    return hand_value + hole_card_value

def is_flush(suits: List[str]) -> bool:
    return any(len(set(hand)) == 1 for hand in itertools.combinations(suits, 5))

def is_straight(sorted_ranks: List[int]) -> bool:
    return any(hand == tuple(range(min(hand), max(hand) + 1)) 
              for hand in itertools.combinations(sorted_ranks, 5))

def is_four_of_a_kind(ranks: List[int]) -> bool:
    return any(any(hand.count(rank) == 4 for rank in hand)
              for hand in itertools.combinations(ranks, 5))

def is_full_house(ranks: List[int]) -> bool:
    return any(is_three_of_a_kind(hand) and is_one_pair(hand)
              for hand in itertools.combinations(ranks, 5))

def is_three_of_a_kind(ranks: List[int]) -> bool:
    return any(any(hand.count(rank) == 3 for rank in hand)
              for hand in itertools.combinations(ranks, 5))

def is_two_pair(ranks: List[int]) -> bool:
    return any(sum(1 for rank in set(hand) if hand.count(rank) == 2) == 2
              for hand in itertools.combinations(ranks, 5))

def is_one_pair(ranks: List[int]) -> bool:
    return any(any(hand.count(rank) == 2 for rank in hand)
              for hand in itertools.combinations(ranks, 5)) 