import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import imageio
import ale_py
import os
from collections import deque
import argparse


class DQN(nn.Module):
    def __init__(self, input_shape=(4,), num_actions=2, network_type="cnn"):
        super(DQN, self).__init__()
        self.network_type = network_type

        if network_type == "cnn":
            # CNN architecture for Atari games
            self.network = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
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
            x = x / 255.0
        return self.network(x)


class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(
            self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame.copy())
        stacked = np.stack(self.frames, axis=0)
        return stacked


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Determine environment and network type based on args
    if args.env_name.startswith("ALE"):
        env = gym.make(args.env_name, render_mode="rgb_array")
        network_type = "cnn"
        preprocessor = AtariPreprocessor()
        input_shape = (4, 84, 84)  # For Atari
    else:
        env = gym.make(args.env_name, render_mode="rgb_array")
        network_type = "mlp"
        preprocessor = None
        input_shape = env.observation_space.shape

    env.action_space.seed(args.seed)
    if hasattr(env.observation_space, 'seed'):
        env.observation_space.seed(args.seed)

    num_actions = env.action_space.n
    print(f"Environment: {args.env_name}, Network type: {network_type}")
    print(f"Input shape: {input_shape}, Number of actions: {num_actions}")

    model = DQN(input_shape=input_shape, num_actions=num_actions,
                network_type=network_type).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        if network_type == "cnn":
            state = preprocessor.reset(obs)
        else:
            state = obs

        done = False
        total_reward = 0
        frames = []
        frame_idx = 0

        while not done:
            frame = env.render()
            frames.append(frame)

            state_tensor = torch.from_numpy(
                np.array(state)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if network_type == "cnn":
                state = preprocessor.step(next_obs)
            else:
                state = next_obs

            frame_idx += 1

        # Create episode directory for frames if needed
        frames_dir = os.path.join(args.output_dir, f"episode_{ep}_frames")

        # Try to save as video first (requires ffmpeg)
        try:
            out_path = os.path.join(args.output_dir, f"eval_ep{ep}.mp4")
            with imageio.get_writer(out_path, fps=30) as video:
                for f in frames:
                    video.append_data(f)
            print(
                f"Saved episode {ep} with total reward {total_reward} → {out_path}")
        except ValueError as e:
            # Fallback: save individual frames as PNG
            print(f"Could not create video: {e}")
            print(f"Saving individual frames instead.")
            os.makedirs(frames_dir, exist_ok=True)
            for i, f in enumerate(frames):
                frame_path = os.path.join(frames_dir, f"frame_{i:05d}.png")
                imageio.imwrite(frame_path, f)
            print(
                f"Saved {len(frames)} frames from episode {ep} with total reward {total_reward} to {frames_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        required=True, help="Path to trained .pt model")
    parser.add_argument("--env-name", type=str,
                        default="CartPole-v1", help="Environment name")
    parser.add_argument("--output-dir", type=str, default="./eval_videos")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=313551076,
                        help="Random seed for evaluation")
    args = parser.parse_args()
    evaluate(args)
