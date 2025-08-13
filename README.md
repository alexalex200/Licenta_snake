# AI in the Snake Game: Genetic Algorithms and Reinforcement Learning (PPO)

This project is my bachelor's thesis and aims to provide a comparative analysis of two artificial intelligence methods applied to the Snake game: **Genetic Algorithms** and **Reinforcement Learning** using **Proximal Policy Optimization (PPO)**.

## 📌 Objective
The goal is to develop agents capable of learning to play Snake efficiently, with the ultimate aim of **completely filling the game board**.
Both methods use neural networks for decision-making, but they differ fundamentally in their optimization approach:
- **Genetic Algorithm** evolves a population of individuals based on a fitness score.
- **Reinforcement Learning (PPO)** updates its policy through direct interaction with the environment.

## 🐍 Snake Game Description
- The player (agent) controls a snake on a 2D grid.
- The goal is to collect apples while avoiding collisions with walls or the snake’s own body.
- Each apple eaten increases the snake's length.

Evaluation metrics:
- **Total score** – number of apples collected.
- **Number of steps** – efficiency in finishing the game.
- **Complete game finish** – filling the entire board.

## 🧠 Neural Network Architecture
- **Input:** 32-element vector containing distances to obstacles, apple positions, and movement direction.
- **Two hidden layers:** 20 and 12 neurons, ReLU activation.
- **Output:** 3 neurons (turn left, move forward, turn right) with Softmax activation.

## 🔬 Implemented Methods

### 1. Genetic Algorithm
- Evolves a population of agents via selection, crossover, and mutation.
- Fitness is based on steps survived and apples collected.
- Advantages: works without explicit rewards, encourages diversity.
- Limitations: high computational cost, slow convergence.

### 2. Reinforcement Learning (PPO)
- Actor-Critic with two separate networks: one for actions (actor) and one for state value estimation (critic).
- On-policy learning, stable updates via **clipping** mechanism.
- Advantages: high adaptability, consistent performance.
- Limitations: sensitive to hyperparameters, higher training cost.

## 📊 Results
- **PPO** showed the best consistency and efficiency in completing the game.
- **Genetic Algorithm** had competitive performance but lower adaptability.
- **Naive solution** always finishes the game but is inefficient in terms of steps.

## 🛠️ Technologies Used
- **Python**
- [PyTorch](https://pytorch.org/) – for neural networks and training.
- Custom implementation of Genetic Algorithms.
- PPO based on policy gradient.

**Author:** Anca Alexandru-Iulian  
**Scientific Coordinator:** Prof. Dr. Alexe Bogdan  
Faculty of Mathematics and Computer Science, University of Bucharest – 2025
