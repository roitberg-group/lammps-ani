import pandas as pd
import matplotlib.pyplot as plt

# Load the strong scaling data from the CSV file
df_strong_scaling = pd.read_csv('../all.csv')  # update this with the path to your CSV file

# Separate the data by atom count
df_300k = df_strong_scaling[df_strong_scaling['atoms'] == 300003]
df_1m = df_strong_scaling[df_strong_scaling['atoms'] == 1000002]
df_10m = df_strong_scaling[df_strong_scaling['atoms'] == 10000002]

# Define the color scheme
colors = ['red', 'blue', 'green']

fig, axs = plt.subplots(2, 1, figsize=(5, 7), dpi=150)

# First subplot for Timesteps/s vs Number of GPUs
axs[0].plot(df_300k['num_gpus'], df_300k['timesteps/s'], 'o-', color=colors[0], label='300k atoms')
axs[0].plot(df_1m['num_gpus'], df_1m['timesteps/s'], 'o-', color=colors[1], label='1M atoms')
axs[0].plot(df_10m['num_gpus'], df_10m['timesteps/s'], 'o-', color=colors[2], label='10M atoms')

# Add labels and title
axs[0].set_xlabel('Number of GPUs')
axs[0].set_ylabel('Timesteps/s')
axs[0].set_title('Strong Scaling: Timesteps/s vs Number of GPUs')

# Set axes to log scale
axs[0].set_xscale('log')
axs[0].set_yscale('log')

# Add a grid
axs[0].grid(True)

# Add a legend
axs[0].legend()

# Second subplot for Matoms_step/s vs Number of GPUs
axs[1].plot(df_300k['num_gpus'], df_300k['Matoms_step/s'], 'o-', color=colors[0], label='300k atoms')
axs[1].plot(df_1m['num_gpus'], df_1m['Matoms_step/s'], 'o-', color=colors[1], label='1M atoms')
axs[1].plot(df_10m['num_gpus'], df_10m['Matoms_step/s'], 'o-', color=colors[2], label='10M atoms')

# Add labels and title
axs[1].set_xlabel('Number of GPUs')
axs[1].set_ylabel('Matoms_step/s')
axs[1].set_title('Strong Scaling: Matoms_step/s vs Number of GPUs')

# Set axes to log scale
axs[1].set_xscale('log')
axs[1].set_yscale('log')

# Add a grid
axs[1].grid(True)

# Add a legend
axs[1].legend()

plt.tight_layout()
plt.show()
plt.savefig("strong_scale.png")
