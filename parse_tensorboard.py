import os
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def plot_rewards_from_tb(log_dir="runs"):
    """
    Plots rewards from a TensorBoard log directory.
    """
    # find the latest run directory
    try:
        run_dirs = sorted(
            [
                os.path.join(log_dir, d)
                for d in os.listdir(log_dir)
                if os.path.isdir(os.path.join(log_dir, d))
            ]
        )
    except FileNotFoundError:
        print(f"Log directory '{log_dir}' not found. Run training first.")
        return

    if not run_dirs:
        print(f"No run directories found in '{log_dir}'.")
        return

    latest_run_dir = run_dirs[-1]
    print(f"Loading data from: {latest_run_dir}")

    # find the event file
    event_file = None
    for f in os.listdir(latest_run_dir):
        if f.startswith("events.out.tfevents"):
            event_file = os.path.join(latest_run_dir, f)
            break

    if not event_file:
        print(f"No event file found in '{latest_run_dir}'.")
        return

    # load data using EventAccumulator
    ea = event_accumulator.EventAccumulator(
        event_file, size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()

    try:
        episode_rewards = pd.DataFrame(ea.Scalars("Reward/episode")).rename(
            columns={"value": "Episode Reward", "step": "Episode"}
        )
        avg_rewards = pd.DataFrame(ea.Scalars("Reward/avg")).rename(
            columns={"value": "Average Reward", "step": "Episode"}
        )
    except KeyError as e:
        print(
            f"Could not find scalar tag: {e}. Make sure training has logged these values."
        )
        return

    # Plotting
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(12, 7))

    plt.plot(
        episode_rewards["Episode"],
        episode_rewards["Episode Reward"],
        label="Episode Reward",
        alpha=0.6,
        color="C0",
    )
    plt.plot(
        avg_rewards["Episode"],
        avg_rewards["Average Reward"],
        label="Average Reward (100 episodes)",
        linewidth=2.5,
        color="C1",
    )

    plt.title("REINFORCE Training on CartPole-v1", fontsize=16)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Save the plot
    plot_filename = "reward_plot.png"
    plt.savefig(plot_filename)
    print(f"Plot saved successfully to '{plot_filename}'.")


if __name__ == "__main__":
    plot_rewards_from_tb()
