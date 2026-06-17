"""
Tabular Q-Learning on FrozenLake-vq.

Q-Learning update rule:
    Q(s,a) <- Q(s,a) + alpha * [r + y * max_a'Q(s',a') - Q(s,a)]

The term in brackets is the TD error: 
    "how wrong was our current estimate of Q(s,a)?"

If we got reward r and landed in state s', the best we could do from s'
is max_a'Q(s',a'). So the true value of (s,a) should be r + y*max_a'Q(s',a').
We nudge our estimate toward that truth by step size alpha.
"""

import numpy as np
import gymnasium as gym
from dataclasses import dataclass

@dataclass 
class QLearningConfig:
    alpha: float = 0.8   #learning rate - how fast to update Q values
    gamma: float = 0.95  #discount factor - how much future rewards matter
    epsilon: float = 1.0  #exploration rate - start fully random
    epsilon_min: float = 0.01   #never go fully greedy
    epsilon_decay: float = 0.001   #how fast to reduce exploration
    episodes: int = 10_000   #total training episodes
    eval_interval: int = 1_000   #evaluate every N episodes



def train_q_learning(config: QLearningConfig) -> tuple[np.ndarray, list[float]]:
    """
    Train a Q-table on FrozenLake.

    Returns:
        Q: trained Q-table, shape (n_states, n_actions) = (16, 4)
        rewards_per_episode: list of total reward per episode
    """
    env = gym.make("FrozenLake-v1", is_slippery=True)

    n_states = env.observation_space.n   #16
    n_actions = env.action_space.n   #4

    #Initialize Q-table to zeros
    #Q[s,a] = estimated total future reward from state s taking action a
    Q = np.zeros((n_states, n_actions))

    rewards_history = []

    for episode in range(config.episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            #Epsilon-greedy action selection
            #With probability episilon: explore (random action)
            #With probability 1-epsilon: exploit (best known action)
            if np.random.random() < config.epsilon:
                action = env.action_space.sample()    #explore
            else:
                action = np.argmax(Q[state])    #exploit

            #Take action, observe result
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            #Q-learning update
            best_next = np.max(Q[next_state])   #max_a'Q(s',a')
            td_target = reward + config.gamma * best_next * (not done) 
            td_error = td_target - Q[state, action]
            Q[state, action] += config.alpha * td_error

            state = next_state
            total_reward += reward

        #Decay epsilon - explore less as we learn more
        config.epsilon = max(
            config.epsilon_min,
            config.epsilon - config.epsilon_decay
        )

        rewards_history.append(total_reward)

        #Evaluation
        if (episode + 1) % config.eval_interval == 0:
            recent_wins = sum(rewards_history[-500:]) / 500 
            print(
                f"Episode {episode+1:6d} | "
                f"Win rate (last 500): {recent_wins:.2%} | "
                f"Epsilon: {config.epsilon:.3f}"
            )

    env.close()
    return Q, rewards_history


def evaluate_policy(Q: np.ndarray, n_episodes: int = 1000) -> float:
    """
    Run the greedy policy (no exploration) and return win rate.
    This is the true test - no randomness, pure learned behavior.
    """
    env = gym.make("FrozenLake-v1", is_slippery=True)
    wins = 0

    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = np.argmax(Q[state])   #always take best known action
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if reward == 1.0:
                wins += 1

    env.close()
    return wins / n_episodes

if __name__ == "__main__":
    np.random.seed(42)
    config = QLearningConfig()

    print("Training Q-Learning agent on FrozenLake...")
    print("-" * 55)

    Q, history = train_q_learning(config)


    win_rate = evaluate_policy(Q)
    print(f"\nFinal evaluation over 1000 episodes: {win_rate:.2%} win rate")

    print("\nLearned Q-Table (rows=states 0-15, cols=L/D/R/U):")
    print(np.round(Q, 3))

    if win_rate > 0.70:
        print("\nPASSED - agent achieves >70% win rate")
    else:
        print(f"\nNot there yet - win rate {win_rate:.2%}, target >70%")


