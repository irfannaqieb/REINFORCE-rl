# REINFORCE for CartPole-v1

This repository contains a simple implementation of the REINFORCE algorithm to solve the `CartPole-v1` environment from [Gymnasium](https://gymnasium.farama.org/).

## Requirements

The dependencies are listed in `requirements.txt`. You can install them using pip:

```bash
pip install -r requirements.txt
```

## Usage

To train the agent, run the following command:

```bash
python reinforce_cartpole.py
```

The script will train the agent and print the progress. It will stop once the environment is solved or the maximum number of episodes is reached.

## Implementation Details

-   **Algorithm:** REINFORCE (Monte Carlo Policy Gradient)
-   **Environment:** `CartPole-v1`
-   **Framework:** PyTorch
-   **Policy Network:** A simple Multi-Layer Perceptron (MLP) with one hidden layer.
