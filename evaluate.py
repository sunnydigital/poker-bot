import random
from typing import List, Optional, Dict, Any
import matplotlib.pyplot as plt
from poker_agent import PokerAgent
from random_agent import RandomAgent
from game_state import GameState

def simulate_game(agents: List[PokerAgent], num_rounds: int = 1000) -> Dict[str, List[int]]:
    """
    Simulate multiple rounds of poker between different agents.
    
    Args:
        agents: List of poker agents to compete
        num_rounds: Number of rounds to simulate
        
    Returns:
        Dict containing the win history for each agent
    """
    game_state = GameState()
    win_history = {f"Agent_{i}": [] for i in range(len(agents))}
    wins = {f"Agent_{i}": 0 for i in range(len(agents))}
    
    for round_num in range(num_rounds):
        game_state.reset()
        current_player = 0
        done = False
        
        while not done:
            agent = agents[current_player]
            # Get game state information
            community_cards = game_state.get_community_cards()
            player_hand = game_state.get_player_hand(current_player)
            pot = game_state.get_pot()
            current_bet = game_state.get_current_bet()
            
            # Get agent's action
            if isinstance(agent, PokerAgent):  # LLM agent
                action_dict = agent.get_action(game_state)
                action = action_dict["action"]
            else:  # Other agents (Random, etc.)
                action = agent.calculate_action(
                    community_cards=community_cards,
                    player_hand=player_hand,
                    pot=pot,
                    current_bet=current_bet
                )
            
            # Apply action and get new state
            done = game_state.apply_action(current_player, action)
            
            if not done:
                current_player = (current_player + 1) % len(agents)
        
        # Record winner
        winner = game_state.get_winner()
        wins[f"Agent_{winner}"] += 1
        
        # Update win history
        for agent_id in win_history:
            win_history[agent_id].append(wins[agent_id])
            
        if (round_num + 1) % 100 == 0:
            print(f"Completed {round_num + 1} rounds")
            
    return win_history

def plot_results(win_history: Dict[str, List[int]], agent_names: List[str]):
    """Plot the cumulative wins for each agent."""
    plt.figure(figsize=(10, 6))
    for agent_id, wins in win_history.items():
        agent_index = int(agent_id.split('_')[1])
        plt.plot(range(1, len(wins) + 1), wins, label=agent_names[agent_index])
    
    plt.xlabel('Number of Rounds')
    plt.ylabel('Cumulative Wins')
    plt.title('Agent Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Create agents
    random_agent = RandomAgent()
    llm_agent = PokerAgent(model_name="gpt-4")  # 使用GPT-4模型
    
    # 添加多个代理进行对抗
    agents = [random_agent, llm_agent]
    agent_names = ["Random Agent", "LLM Agent"]
    
    # 模拟对战
    win_history = simulate_game(agents, num_rounds=100)  # 可以调整轮数
    
    # 绘制结果
    plot_results(win_history, agent_names) 