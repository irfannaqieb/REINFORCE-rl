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

## Visualizing Results

This project uses TensorBoard to log and visualize the training progress. To view the logs, run the following command in your terminal:

```bash
tensorboard --logdir=runs
```

This will start a web server that you can navigate to in your browser to see the reward curves.

### Generating a Plot

You can also generate a static plot of the training rewards using Matplotlib. After running the training script, execute the following command:

```bash
python plot_results.py
```

This will read the data from the latest TensorBoard log and create a `reward_plot.png` file in the root directory.
