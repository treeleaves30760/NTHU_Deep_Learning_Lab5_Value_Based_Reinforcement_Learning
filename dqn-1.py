# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time
import datetime

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """

    def __init__(self, num_actions, input_shape=(4,), network_type="mlp"):
        super(DQN, self).__init__()
        self.network_type = network_type

        if network_type == "cnn":
            # CNN architecture for Atari games
            self.network = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions)
            )
        elif network_type == "mlp":
            # MLP architecture for CartPole and other simple environments
            self.input_dim = input_shape[0]
            self.network = nn.Sequential(
                nn.Linear(self.input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, num_actions)
            )

    def forward(self, x):
        # Simple normalization for CartPole
        if self.network_type == "mlp":
            x = x.view(-1, self.input_dim)
        elif self.network_type == "cnn":
            # Normalize pixel values for CNN
            x = x / 255.0
        return self.network(x)


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """

    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        # Only convert to grayscale if obs has 3 channels (RGB)
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs  # Already grayscale or not an image
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame.copy() for _ in range(
            self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame.copy())  # Use copy to avoid reference issues
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """

    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition, error):
        # Calculate priority
        priority = (abs(error) + 1e-6) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        return

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        # Importance-sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        return samples, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + 1e-6) ** self.alpha
        return

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        # For Atari, use NoFrameskip version of the environment
        if "NoFrameskip" in env_name:
            self.env = gym.make(env_name, render_mode="rgb_array")
            self.test_env = gym.make(env_name, render_mode="rgb_array")
        else:
            self.env = gym.make(env_name, render_mode="rgb_array")
            self.test_env = gym.make(env_name, render_mode="rgb_array")

        self.num_actions = self.env.action_space.n
        self.env_name = env_name

        # Choose network type based on environment
        if "CartPole" in env_name:
            self.network_type = "mlp"
            # Get input shape from the environment
            self.input_shape = self.env.observation_space.shape
            # For CartPole, use a smaller replay buffer and start training sooner
            self.replay_start_size = min(args.replay_start_size, 1000)
            self.clip_rewards = False
        else:
            # Atari or other visual environments
            self.network_type = "cnn"
            self.input_shape = (4, 84, 84)  # Standard for Atari with 4 frames
            self.preprocessor = AtariPreprocessor()
            self.replay_start_size = args.replay_start_size
            self.clip_rewards = True  # Clip rewards for Atari

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        print(f"Using {self.network_type} network for {env_name}")

        self.q_net = DQN(self.num_actions, self.input_shape,
                         self.network_type).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(
            self.num_actions, self.input_shape, self.network_type).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        # Faster epsilon decay for CartPole
        if "CartPole" in env_name:
            self.epsilon_decay = 0.995  # Much faster decay for CartPole
        else:
            self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        # CartPole max reward is 500 (standard success threshold: 475)
        self.best_reward = 0 if "CartPole" in env_name else -21  # Default for Pong
        self.max_episode_steps = args.max_episode_steps
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Beta annealing for prioritized replay
        self.beta_start = 0.4
        self.beta_end = 1.0
        self.beta_frames = 100000
        self.beta = self.beta_start

        self.memory = PrioritizedReplayBuffer(
            args.memory_size, beta=self.beta_start)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(
            np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=1000):
        for ep in range(episodes):
            obs, _ = self.env.reset()

            if self.network_type == "cnn":
                state = self.preprocessor.reset(obs)
            else:
                # For CartPole, no preprocessing needed
                state = obs

            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(
                    action)
                done = terminated or truncated

                # Clip rewards for Atari games (standard practice)
                if self.clip_rewards:
                    reward = np.sign(reward)  # -1, 0, 1

                if self.network_type == "cnn":
                    next_state = self.preprocessor.step(next_obs)
                else:
                    # For CartPole, no preprocessing needed
                    next_state = next_obs

                self.memory.add((state, action, reward, next_state, done), 1.0)

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                # Anneal beta for prioritized replay
                self.beta = min(self.beta_end, self.beta_start + (self.beta_end - self.beta_start) *
                                (self.env_count / self.beta_frames))
                self.memory.beta = self.beta

                if self.env_count % 1000 == 0:
                    print(
                        f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f} Beta: {self.beta:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon,
                        "Beta": self.beta
                    })
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed

                    ########## END OF YOUR CODE ##########
            print(
                f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed

            ########## END OF YOUR CODE ##########
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(
                        f"Saved new best model to {model_path} with reward {eval_reward}")
                print(
                    f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def evaluate(self):
        obs, _ = self.test_env.reset()

        if self.network_type == "cnn":
            state = self.preprocessor.reset(obs)
        else:
            # For CartPole, no preprocessing needed
            state = obs

        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(
                np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(
                action)
            done = terminated or truncated
            total_reward += reward  # Don't clip rewards during evaluation

            if self.network_type == "cnn":
                state = self.preprocessor.step(next_obs)
            else:
                # For CartPole, no preprocessing needed
                state = next_obs

        return total_reward

    def train(self):
        if len(self.memory) < self.replay_start_size:
            return

        # Decay function for epsilon-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1

        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        batch, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.from_numpy(
            np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(
            np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        # Q(s,a)
        q_values = self.q_net(states).gather(
            1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: Use online network to select actions and target network to evaluate them
        with torch.no_grad():
            # Get actions from online network
            argmax_actions = self.q_net(next_states).argmax(1, keepdim=True)
            # Evaluate selected actions using target network
            next_q_values = self.target_net(next_states).gather(
                1, argmax_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q_values * (1 - dones)

        # TD error
        td_errors = q_values - target_q
        # Huber loss with importance-sampling weights
        loss = (weights * torch.nn.functional.smooth_l1_loss(q_values,
                target_q, reduction='none')).mean()
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)

        self.optimizer.step()

        # Update priorities
        self.memory.update_priorities(
            indices, np.abs(td_errors.detach().cpu().numpy()))

        if self.train_count % 1000 == 0:
            print(
                f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")
            wandb.log({
                "Loss": loss.item(),
                "Q_mean": q_values.mean().item(),
                "Q_std": q_values.std().item(),
                "Target_Q_mean": target_q.mean().item()
            })

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str,
                        default="cartpole-run-improved")
    # Larger batch for stable learning
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--memory-size", type=int, default=100000)
    # Higher learning rate for CartPole
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float,
                        default=0.999999)  # Default for Atari
    parser.add_argument("--epsilon-min", type=float,
                        default=0.01)  # Lower epsilon minimum
    parser.add_argument("--target-update-frequency", type=int,
                        default=500)  # More frequent updates
    parser.add_argument("--replay-start-size", type=int,
                        default=50000)  # Default for Atari
    parser.add_argument("--max-episode-steps", type=int,
                        default=500)  # CartPole default max steps
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-name", type=str, default="CartPole-v1")
    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create timestamped directory for saving results
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir

    wandb.init(project="DLP-Lab5-DQN",
               name=f"{args.env_name}-{args.wandb_run_name}",
               save_code=True,
               config=vars(args))

    agent = DQNAgent(env_name=args.env_name, args=args)
    agent.run()
