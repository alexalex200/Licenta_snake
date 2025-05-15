# Create a new file called demo.py
import torch
import pygame
from game import Game
from draw_game import Draw  # Import your game classes
from rl import DQN, SnakeGameEnv, demo_agent  # Import your DQN and environment classes

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")

# Create environment
env = SnakeGameEnv(board_size=(20, 20), num_snakes=1, agent_index=0)

# Initialize model
state, _ = env.reset()
n_observations = len(state)
n_actions = env.action_space.n
policy_net = DQN(n_observations, n_actions).to(device)

# Load the trained model
policy_net.load_state_dict(torch.load("snake_dqn_model.pth"))
policy_net.eval()  # Set to evaluation mode


# Run the demo
demo_agent(policy_net, env, num_episodes=5)