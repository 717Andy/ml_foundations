"""
REINFORCE (Monte Carlo policy Gradient) on CartPole-v1.

Algorithm:
   1. Run full episode using current policy
   2. Compute discounted returns G_t for eavh timestep
   3. Compute policy gradient loss: -sum(log_prob(a_t) * G_t)
   4. Backprop and update policy network weights
   5. Repeat

This is direct connectiom between Phase 0 (neural nets + backprop)
and Phase 1 (RL). The policy IS an MLP. Training IS gradient descent.
The only difference: the loss signal comes from experience, not labels.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from dataclasses import dataclass
from collections import deque


@dataclass
class REINFORCEConfig:
    gamma: float = 0.99   #discount factor
    learning_rate: float = 1e-3
    hidden_dims: list = None   #policy network hidden layers
    max_episodes: int = 2000
    target_reward: float = 475.00   #CartPole solved threshold
    eval_window: int = 100   #rolling window for evaluation
    log_interval: int = 50

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 128]


class PolicyNetwork(nn.Module):
    """
    The agent's brain. Maps state observations to action probabilities.

    This is your Phase 0 MLP - same architecture, different purpose.
    Input: state vector (4 values for CartPole)
    Output: prabability distribution over actions (2 values for Cartpole)

    The softmax at the end ensures outputs sum to 1.0 (valid probabilities)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list[int]):
        super().__init__()

        dims = [state_dim] + hidden_dims + [action_dim]
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())

        #Softmax converts raw output scores to probabilities
        #dim=-1 means "softmax over the last dimension" (the action dimension)
        layers.append(nn.Softmax(dim=-1))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (state_dim,) or (batch, state_dim)
        returns: action probabilities, same leading shape, last dim = action_dim
        """
        return self.network(state)
    

def compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """
    Compute discounted returns G_t for each timestep in an episode.

    G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ... + gamma^2{T-t}*r_T

    We compute this backwards - start from the last reward and 
    accumulate forward. This is O(T) instead of O(T^2).

    Arges:
        rewards: list of rewards [r_0, r_1, ..., r_T]
        gamma: discount factor 

    Returns:
        returns: tensor of shape (T,) - G_t for each timestep
    """ 
    G = 0.0
    returns = []

    #Walk backwards through rewards
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)

    #Normalize returns - crucial for stable training 
    #Prevents early (high-return) episodes from dominating updates
    #Same reason you standardized inputs in the NumPy MLP
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns


def select_action(
        policy: PolicyNetwork,
        state: np.ndarray,        
) -> tuple[int, torch.Tensor]:
    """
    Sample an action from the policy's probability distribution.

    This is stochatic - we SAMPLE, not argmax. This is essential:
    - Argmax would collapse to a dterministic policy immediately
    - Sampling maintains  exploration throughout teaining 
    - the log_prob we save is used in the policy gradient update 

    Returns:
        actions: interger action to take
        log_prob: log probability of this action (saved for gradient computation)
    """
    state_tensor = torch.from_numpy(state).float()

    #Get action probabilities from policy network
    probs = policy(state_tensor)

    #Categorial distribution - samples from a discrete prob distribution
    dist = Categorical(probs)

    #Sample an action
    action = dist.sample()

    #Return action as int and its log probability
    #log_prob will be used in loss computation: -log_prob * G_t
    return action.item(), dist.log_prob(action)


def train_reinforce(config:REINFORCEConfig) -> tuple[PolicyNetwork, list[float]]:
    """
    Full REINFORCE training loop.

    Returns trained policy and episode reward history.
    """
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, action_dim, config.hidden_dims)
    optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate)

    reward_history = []
    recent_rewards = deque(maxlen=config.eval_window)

    print(f"Policy network: {sum(p.numel() for p in policy.parameters()):,} parameters")
    print(f"State dim: {state_dim} | Action dim: {action_dim}")
    print("-" * 60)

    for episode in range(config.max_episodes):
        #Collect one full episode
        state, _ = env.reset()
        log_probs = []   #log pi(a_t | s_t) for each timestep
        rewards = []   #r_t for each timestep
        done = False

        while not done:
            action, log_prob = select_action(policy, state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)

        #Compute returns 
        returns = compute_returns(rewards, config.gamma)
        total_reward = sum(rewards)

        #Compute Policy Gradient loss
        """
        loss = -sum(log_prob(a_t) * G_t)

        Why negative? Because:
        - We MAXIMIZE expected return (RL objective)
        - Optimizers in PyTorch MINIMIZE loss
        - So we negate: minimizing -E[return] = maximizing E[return]

        Why log_prob * G_t?
        - High G_t: this sequence of actions led to good outcomes
          -> large gradient -> policy moves away from these actions
        - Low G_t: bad outcomes -> policy moves away from these actions
        - log_prob: differentiable connection to policy network weights
          -> backprop can flow through this into the network parameters
        """

        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)

        loss = torch.stack(policy_loss).sum()

        #Backprop and update 
        optimizer.zero_grad()
        loss.backward()
        #Clip gradients - REINFORCE can have high variamce: this stabilizes it 
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        #Logging 
        reward_history.append(total_reward)
        recent_rewards.append(total_reward)
        avg_reward = np.mean(recent_rewards)

        if (episode + 1 ) % config.log_interval == 0:
            print(
                f"Episode {episode+1:5d} | "
                f"Reward: {total_reward:6.1f} | "
                f"Avg(last {config.eval_window}): {avg_reward:6.1f} | "
                f"Loss: {loss.item():8.3f}"
            )

        #Check if solved
        if avg_reward >= config.target_reward and len(recent_rewards) == config.eval_window:
            print(f"\n SOLVED at episode {episode+1}!")
            print(f"    Average reward over last {config.eval_window} episodes: {avg_reward:.1f}")
            break

    env.close()
    return policy, reward_history


def evaluate_policy(policy: PolicyNetwork, n_episodes: int = 100) -> float:
    """Evaluate trained policy greedily (no sampling, take highest-prob action)"""
    env = gym.make("CartPole-v1")
    total_rewards = []

    policy.eval()
    with torch.no_grad():
        for _ in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                state_tensor = torch.from_numpy(state).float()
                probs = policy(state_tensor)
                action = torch.argmax(probs).item()   #greedy at eval time
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward

            total_rewards.append(episode_reward)

    env.close()
    return np.mean(total_rewards)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    config = REINFORCEConfig()
    policy, history = train_reinforce(config)

    print("\nRunning final evaluation (greedy policy, 100 episodes)...")
    final_score = evaluate_policy(policy)
    print(f"Final evaluation score: {final_score:.1f} / 500.0")

    #Save the trained policy
    torch.save(policy.state_dict(), "Experiments/cartpole_reinforce.pt")
    print("Policy saved to experiments/cartpole_reinforce.pt")



