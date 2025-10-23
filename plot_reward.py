import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

directory = Path('history/')

# 2. Get a list of all files and find the latest one
# The generator expression (f for f in ...) is memory-efficient
try:
    latest_file = max(directory.glob('*'), key=lambda f: f.stat().st_mtime)
    print(f"The latest file is: {latest_file}")
except ValueError:
    latest_file = None
    print("The directory is empty.")

if latest_file:
    reward_history = pd.read_csv(latest_file)
    reward_history = reward_history[-10000:]
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10, 12))
    for col, ax in zip(reward_history.columns, axes.flatten()):
        ax.plot(reward_history[col])
        ax.set_title(f"Reward History - {col}")
        ax.set_xlabel('time') # Set the x-axis label
        ax.set_ylabel(col)
    plt.tight_layout()
    plt.show()