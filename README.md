# Deep Q-Learning for Lunar Landing

This project implements a Deep Q-Learning (DQN) agent to solve the LunarLander-v2 environment from OpenAI Gymnasium.

## Gymnasium

Gymnasium is a toolkit for developing and comparing reinforcement learning algorithms. It provides a wide range of environments for training agents, including classic Atari games, board games, and physics simulations like LunarLander-v2.

## Deep Q-Learning

Deep Q-Learning is a popular reinforcement learning technique that uses a neural network (NN) to approximate the Q-values of state-action pairs. It combines Q-Learning with deep learning to handle high-dimensional state spaces.

## Project Structure

- **agent.py**: Contains the implementation of the DQN agent.
- **network.py**: Defines the neural network architecture.
- **replay_memory.py**: Implements the replay memory for experience replay.
- **train.py**: Trains the DQN agent using the LunarLander-v2 environment.
- **utils.py**: Provides utility functions for video rendering and visualization.

## Requirements

- Python 3.x
- Gymnasium
- PyTorch
- NumPy
- OpenAI Gym

## Usage

1. Install the required packages: `pip install -r requirements.txt`
2. Run `python train.py` to train the DQN agent.
3. Run `python test.py` to test the trained agent.
4. Use `show_video()` function in `utils.py` to visualize the agent's performance.

## Results

The agent successfully solves the LunarLander-v2 environment, achieving an average score of over 200 in 100 consecutive episodes.

## References

- [Deep Q-Learning with PyTorch](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [OpenAI Gym Documentation](https://gym.openai.com/docs/)
