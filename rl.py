import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import gym
from gym import spaces
import numpy as np
from game import Game
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import pygame
from draw_game import Draw

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class SnakeGameEnv(gym.Env):
    def __init__(self, board_size=(20, 20), num_snakes=1, agent_index=0):
        super(SnakeGameEnv, self).__init__()

        # Initialize your existing game
        self.game = Game(board_size=board_size, num_snakes=num_snakes)

        # Which snake the agent controls
        self.agent_index = agent_index

        # Define action space (4 directions)
        # 0: up, 1: left, 2: down, 3: right
        self.action_space = spaces.Discrete(4)

        # Define observation space based on your preprocess_state method
        # Your state has 16 elements (2 for position, 4 for direction, 8 for food, 4 for danger)
        self.observation_space = spaces.Box(
            low=0, high=np.inf,
            shape=(16,), dtype=np.float32
        )

        # Track previous state for reward calculation
        self.previous_score = 0

    def reset(self):
        # Reset game
        self.game.reset()

        # Reset score tracking
        self.previous_score = 0

        # Get initial state
        state = self.game.preprocess_state(self.agent_index)
        info = {}

        return state, info

    def step(self, action):
        # Convert action (0,1,2,3) to direction format used by your game
        directions = [
            (0, -1),  # Up
            (-1, 0),  # Left
            (0, 1),  # Down
            (1, 0)  # Right
        ]

        direction = directions[action]

        # Change direction of the agent's snake
        self.game.change_direction(self.game.snakes[self.agent_index], direction)

        # Update game state
        self.game.update()

        # Get new state
        state = self.game.preprocess_state(self.agent_index)

        # Calculate reward based on score difference
        current_score = self.game.snakes[self.agent_index].score if not self.game.snakes[self.agent_index].dead else 0
        reward = current_score - self.previous_score
        self.previous_score = current_score

        # Check if done
        done = self.game.snakes[self.agent_index].dead

        # Info dictionary can contain additional information
        info = {"score": current_score}

        return state, reward, done, False, info  # False is for truncated parameter in newer gym versions


# Create the environment
env = SnakeGameEnv(board_size=(20, 20), num_snakes=1, agent_index=0)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def demo_agent(model, env, num_episodes=5, fps=10):
    """
    Run the trained agent for visualization using your existing Draw class
    """
    # Initialize pygame if not already initialized
    if not pygame.get_init():
        pygame.init()

    # Create a Draw instance with the game
    drawer = Draw(env.game)

    for episode in range(num_episodes):
        # Reset the environment
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        done = False
        steps = 0

        print(f"Episode {episode + 1}")

        while not done:
            # Process pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Select action with highest Q-value (no exploration)
            with torch.no_grad():
                action = model(state).max(1).indices.view(1, 1)

            # Take action
            observation, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward
            steps += 1

            # Update state
            if not terminated:
                state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Draw the current game state
            drawer.draw()

            # Display current info
            pygame.display.set_caption(
                f"Snake DQN - Episode: {episode + 1} Step: {steps} Score: {env.game.snakes[env.agent_index].score:.1f}")

        print(f"Episode {episode + 1} finished with reward {total_reward:.1f} after {steps} steps")

        # Pause between episodes
        if episode < num_episodes - 1:
            # Display "Next episode starting..." message
            font = pygame.font.SysFont(None, 36)
            text = font.render(f"Episode complete! Next episode in 3 seconds...", True, (0, 0, 0))
            text_rect = text.get_rect(center=(drawer.width // 2, drawer.height // 2))
            drawer.screen.blit(text, text_rect)
            pygame.display.update()

            # Wait 3 seconds
            pygame.time.wait(3000)

    pygame.quit()

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
torch.save(policy_net.state_dict(), "snake_dqn_model.pth")
plot_durations(show_result=True)
plt.ioff()
plt.show()