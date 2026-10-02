"""
PPO (Proximal Policy Optimization) on CartPole-v1.

Key improvements over REINFORCE:
1. Clipped objective - prevents destructively large policy updates
2. Advantage estimates - lower variance gradient signal via a critic network
3. Multiple epochs    - reuses each batch of experience for K gradient steps
                        (carefully, within the clip constraint)

This is the algorithm at the core of Tesla Optimus and FSD training
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from dataclasses import dataclass, field
from collections import deque


@dataclass
class PPOConfig:
    #Enviroment
    env_name:      str = "CartPole-v1"

    #Network
    hidden_dims:   list = field(default_factory=lambda: [64, 64])

    #PPO core hyperparameters
    clip_epsilon:  float = 0.2    #clipping range for policy ratio
    gamma:         float = 0.99   #discount factor
    gae_lambda:    float = 0.95   #GAE smoothing (explained below)
    entropy_coef:  float = 0.01   #encourages exploration
    value_coef:    float = 0.5    #scales critic loss relative to actor loss

    #Training
    learning_rate: float = 3e-4
    n_steps:       int = 512      #steps collected per update 
    n_epochs:      int = 10       #gradient steps per batch of experience 
    batch_size:    int = 64       #minibatch size within each epoch
    max_updates:   int = 200      #total number of update cycles
    target_reward: float = 475.0
    eval_window:   int = 20       #rolling window over update cycles 



class ActorCritic(nn.Module):
    """
    Combined actor-critic network.

    Shared trunk -> two heads:
      Actor head: outputs action probabilities (the policy π)
      Critic head: outputs state value estimate V(s) - a singular scalar 

    Sharing the trunk is efficient - both heads benefit frome the same
    learned state representation. This is standard in modern RL
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list[int]):
        super().__init__()

        #shared feature extractor 
        trunk_layers = []
        dims = [state_dim] + hidden_dims
        for i in range(len(dims) - 1):
            trunk_layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.Tanh(),   #Tanh preferred over ReLU in PPO - bounded gradients
            ])
        self.trunk = nn.Sequential(*trunk_layers)

        #Actor head - outputs logits (raw scores before softmax)
        self.actor_head = nn.Linear(hidden_dims[-1], action_dim)

        #Critic head - outputs state value estimate
        self.critic_head = nn.Linear(hidden_dims[-1], 1)

        #initialize weights with small values for stable early training
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        #Output heads need smaller init for stability
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: raw action scores, shape (..., action_dim)
            value:  state value estimate, shape (..., 1)
        """
        features = self.trunk(state)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value

    def get_action_and_value(
        self, state: torch.Tensor, action: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log_prob, entropy, value.
        If action is provided, evaluate that action instead of sampling.
        Used during both rollout collection and training.
        """
        logits, value = self.forward(state)
        dist          = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()  #H(π) = -Σ π(a) log π(a)
                                  #maximizing entropy encourages exploration
                                    
        return action, log_prob, entropy, value.squeeze(-1)


def compute_gae(
    rewards:    list[float],
    values:     list[float],
    dones:      list[bool],
    last_value: float,
    gamma:      float, 
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimation (GAE).

    A smarter way to compute advantages that balances:
      - Low bias (long rollouts give accurate return estimates)
      - Low variance (short rollouts are less noisy)

    gae_lambda controls the trade-off:
      λ=0: A_t = r_t + γV(s_{t+1}) - V(s_t)  [low variance, high bias]
      λ=1: A_t = G_t - V(s_t)                [low bias, high variance]
      λ=0.95: standard — best of both worlds

    The recursive formula:
      δ_t   = r_t + γ * V(s_{t+1}) * (1-done) - V(s_t)   [TD error]
      A_t   = δ_t + γλ * A_{t+1}
    """
    advantages = []
    gae        = 0.0

    #Walk backwards through the rollout
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else last_value
        next_done  = dones[t]

        #TD error: how wrong was the critic at this step?
        delta = rewards[t] + gamma * next_value * (1.0 - next_done) - values[t]

        #Accumulate discounted TD errors
        gae   = delta + gamma * gae_lambda * (1.0 - next_done) * gae
        advantages.insert(0, gae)

    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns    = advantages + torch.tensor(values[:len(rewards)], dtype=torch.float32)

    return advantages, returns


def collect_rollout(
    env:    gym.Env,
    policy: ActorCritic,
    config: PPOConfig,
    device: torch.device,
) -> dict:
    """
    Collect n_steps of experience using the current policy.
    This is the data PPO will train on for the next K epochs
    """
    states, actions, log_probs = [], [], []
    rewards, values, dones     = [], [], []

    state, _= env.reset()

    for _ in range(config.n_steps):
        state_tensor = torch.FloatTensor(state).to(device)

        with torch.no_grad():
            action, log_prob, _, value = policy.get_action_and_value(state_tensor)

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        states.append(state_tensor)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(float(reward))
        values.append(value.item())
        dones.append(float(done))

        state = next_state if not done else env.reset()[0]

    #Get value estimate for the final state (needed for GAE)
    with torch.no_grad():
        _, _, _, last_value = policy.get_action_and_value(
            torch.FloatTensor(state).to(device)
        )

    #Compute advantages using GAE
    advantages, returns = compute_gae(
        rewards, values, dones,
        last_value.item(), config.gamma, config.gae_lambda
    )

    return {
        "states": torch.stack(states),
        "actions": torch.stack(actions),
        "log_probs": torch.stack(log_probs),
        "advantages": advantages,
        "returns": returns,
    }


def ppo_update(
    policy:    ActorCritic,
    optimizer: optim.Optimizer,
    rollout:   dict,
    config:    PPOConfig,
    device:    torch.device,
) -> dict[str, float]:
    """
    The PPO update - the heart of the algorithm.

    For n_epochs:
      1. Shuffle rollout data into minibatches
      2. Recompute log_probs and values under CURRENT policy
      3. Compute clipped surrogate loss
      4. Compute critic (value) loss 
      5. Add entropy bonus
      6. Backprop and clip gradients
    """
    states       = rollout["states"].to(device)
    actions      = rollout["actions"].to(device)
    old_logprobs = rollout["log_probs"].to(device).detach()
    advantages   = rollout["advantages"].to(device)
    returns      = rollout["returns"].to(device)

    #Normalize advantages within this batch 
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0, "approx_kl": 0}
    n_updates = 0

    for _ in range(config.n_epochs):
        #Shuffle indices for minibatch sampling
        indices = torch.randperm(config.n_steps)

        for start in range(0, config.n_steps, config.batch_size):
            batch_idx = indices[start: start + config.batch_size]

            #Recompute action probabilities and values under current policy
            _, new_logprobs, entropy, new_values = policy.get_action_and_value(
                states[batch_idx], actions[batch_idx]
            )

            #Policy ratio: π_new(a|s) / π_old(a|s)
            #In log space: exp(log_π_new - log_π_old)
            log_ratio = new_logprobs - old_logprobs[batch_idx]
            ratio     = log_ratio.exp()

            #Approximate KL divergence - used to monitor update size 
            approx_kl = ((ratio - 1) - log_ratio).mean().item()

            batch_advantages = advantages[batch_idx]

            #PPO Clipped Objective 
            #Standard policy gradient term
            pg_loss1 = -batch_advantages * ratio

            #Clipped version - ratio constrained to [1-ε, 1+ε]
            pg_loss2 = -batch_advantages * torch.clamp(
                ratio,
                1 - config.clip_epsilon,
                1 + config.clip_epsilon
            )

            #Take the pessimistic (maximum loss) of the two
            #This is the "proximal" part of PPO
            policy_loss = torch.max(pg_loss1, pg_loss2).mean()

            #Critic (Value) Loss
            #Simple MSE between predicted values and actual returns
            value_loss = nn.functional.mse_loss(new_values, returns[batch_idx])

            #Entropy Bonus
            #Encourages the policy to remain exploratory
            #Without this, policy collapses to deterministic too early
            entropy_loss = -entropy.mean()

            #Combined Loss
            loss = (
                policy_loss
                + config.value_coef * value_loss
                + config.entropy_coef * entropy_loss
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy"] += (-entropy_loss.item())
            metrics["approx_kl"] += approx_kl
            n_updates += 1

    #Average metrics over all minibatch updates
    return {k: v / n_updates for k, v in metrics.items()}


def evaluate(policy: ActorCritic, env_name: str, n_episodes: int = 20) -> float:
    """Greedy evaluation - no sampling, take highest probability action"""
    env = gym.make(env_name)
    policy.eval()
    total_rewards = []

    with torch.no_grad():
        for _ in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                logits, _ = policy(torch.FloatTensor(state))
                action = torch.argmax(logits).item()
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward

            total_rewards.append(episode_reward)

        env.close()
        policy.train()
        return float(np.mean(total_rewards))


def train_ppo(config:PPOConfig) -> tuple[ActorCritic, list[float]]:
    device = torch.device("cpu")
    env = gym.make(config.env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = ActorCritic(state_dim, action_dim, config.hidden_dims).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate, eps=1e-5)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"ActorCritic: {n_params:,} parameters")
    print(f"Collecting {config.n_steps} steps per update | "
          f"{config.n_epochs} epochs per update | "
          f"Batch size {config.batch_size}")
    print("-" * 65)

    reward_history = []
    recent_rewards = deque(maxlen=config.eval_window)

    for update in range(1, config.max_updates + 1):

        #Collect rollout with current policy
        rollout = collect_rollout(env, policy, config, device)

        #Update policy using PPO objective
        metrics = ppo_update(policy, optimizer, rollout, config, device)

        #Evaluate current policy
        avg_reward = evaluate(policy, config.env_name, n_episodes=10)
        reward_history.append(avg_reward)
        recent_rewards.append(avg_reward)
        rolling_avg = np.mean(recent_rewards)

        print(
            f"Update {update:4d} | "
            f"Reward: {avg_reward:6.1f} | "
            f"Avg({config.eval_window}): {rolling_avg:6.1f} | "
            f"KL: {metrics['approx_kl']:.4f} | "
            f"Entropy: {metrics['entropy']:.3f}"
        )

        if rolling_avg >= config.target_reward and len(recent_rewards) == config.eval_window:
            print(f"\n SOLVED at update {update}!")
            print(f"   Rolling avg reward: {rolling_avg:.1f}")
            total_steps = update * config.n_steps
            print(f"   Total enviroment steps: {total_steps:,}")
            break

    env.close()
    torch.save(policy.state_dict(), "experiments/cartpole_ppo.pt")
    print("Policy saved to experiments/cartpole_ppo.pt")
    return policy, reward_history


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    config = PPOConfig()
    policy, history = train_ppo(config)

    print("\nFianl greedy evaluation (100 episodes)...")
    final = evaluate(policy, config.env_name, n_episodes=100)
    print(f"Final Score: {final:.1f} / 500.0")













