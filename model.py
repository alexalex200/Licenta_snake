import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import os
import numpy as np


class Memory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.vals), np.array(
            self.rewards), np.array(self.dones), batches

    def store_memory(self, state, action, prob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []


class Linear_Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lr=0.0005, genetic=False):
        super().__init__()
        self.dimensions = [input_size] + hidden_size + [output_size]
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], output_size),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.genetic = genetic

    def forward(self, x):
        dist = self.actor(x)
        if self.genetic:
            return dist
        dist = Categorical(dist)
        return dist

    def get_values_of_layers(self, x):
        values = []
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                x = layer(x)
                values.append(x.data.cpu().numpy())
                x = F.relu(x)
        return values

    def get_weights_biases(self):
        weights = []
        biases = []
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                weights.append(layer.weight.data.cpu().numpy())
                biases.append(layer.bias.data.cpu().numpy())
        return weights, biases

    def set_weights_biases(self, weights, biases):
        idx = 0
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                layer.weight.data = torch.tensor(weights[idx], dtype=torch.float).to(self.device)
                layer.bias.data = torch.tensor(biases[idx], dtype=torch.float).to(self.device)
                idx += 1

    def save(self, file_name='model.pth'):
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        model_folder_path = './models'
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name))
            self.eval()
        else:
            raise FileNotFoundError(f"Model file {file_name} not found.")


class Critic_Network(nn.Module):
    def __init__(self, input_size, hidden_size, lr=0.0005):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        value = self.critic(x)
        return value

    def save(self, file_name='critic.pth'):
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='critic.pth'):
        model_folder_path = './models'
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name))
            self.eval()
        else:
            raise FileNotFoundError(f"Critic file {file_name} not found.")
