import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
import numpy as np
from game import Game
from model import Linear_Network, Critic_Network, Memory
from draw_game import Draw


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.999, lr=0.0003, gae_lambda=0.95, policy_clip=0.2, batch_size=64,
                 n_epochs=10):

        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = Linear_Network(input_dims, [20, 12], n_actions, lr)
        self.critic = Critic_Network(input_dims, [64, 64], lr)
        self.memory = Memory(batch_size)

    def remember(self, state, action, prob, val, reward, done):
        self.memory.store_memory(state, action, prob, val, reward, done)

    def save_models(self, name="ppo_agent"):
        self.actor.save(name)
        self.critic.save(name)

    def load_models(self, name="ppo_agent"):
        self.actor.load(name)
        self.critic.load(name)

    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float).to(self.actor.device)
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        prob = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, prob, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
                reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device)

            values = torch.tensor(values).to(self.actor.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                # prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                     1 + self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()


if __name__ == "__main__":
    game = Game(board_size=(6, 6), num_apples=1)
    N = 256
    batch_size = 8
    n_epochs = 3
    lr = 0.0005
    agent = Agent(n_actions=3, input_dims=32, batch_size=batch_size, n_epochs=n_epochs, lr=lr)
    n_games = 10000

    best_score = 0
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        game.reset()
        observation = game.get_state()
        done = False
        score = 0

        while not done:
            action, prob, value = agent.choose_action(observation)
            reward, done, score = game.step(action)
            observation_ = game.get_state()
            n_steps += 1
            agent.remember(observation, action, prob, value, reward, done)
            if n_steps % batch_size == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if score >= best_score:
            best_score = score
            agent.save_models(name="ppo_agent")

        print(
            f"Game {i + 1}, Score: {score}, Avg Score: {avg_score}, Best Score: {best_score}, Steps: {n_steps}, Learn Iterations: {learn_iters}")
