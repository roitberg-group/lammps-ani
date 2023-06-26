import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def human_format(num):
    # Function to format numbers to human readable format
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.0f%s' % (num, ['', 'K', 'M', 'B', 'T', 'P'][magnitude])

# Load the strong scaling data from the CSV file
df_strong_scaling = pd.read_csv('all.csv')  # update this with the path to your CSV file

# Get all unique atom counts
atom_counts = df_strong_scaling['atoms'].unique()

# Raise an error if there are more than 10 unique atom counts
if len(atom_counts) > 10:
    raise ValueError("Number of unique atom counts exceeds 10")

# Define a list of 10 colors
colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
markers = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h', 'd', '*']

fig, axs = plt.subplots(2, 1, figsize=(5.5, 6), dpi=150)

# Iterate over the unique atom counts
for i, atom_count in enumerate(atom_counts):
    # Filter the data for the current atom count
    df_atom = df_strong_scaling[df_strong_scaling['atoms'] == atom_count]

    # Plot the Timesteps/s vs Number of GPUs
    axs[0].plot(df_atom['num_gpus'], df_atom['timesteps/s'], marker=markers[i], color=colors[i], label=f'{human_format(atom_count)} atoms')

    # Plot the Matom_step/s vs Number of GPUs
    axs[1].plot(df_atom['num_gpus'], df_atom['Matom_step/s'], marker=markers[i], color=colors[i], label=f'{human_format(atom_count)} atoms')


# Specify the range of your x-axis (modify as needed)
x_min = 0
x_max = int(np.log2(df_atom['num_gpus'].max()))

# Generate the ticks
x_ticks = np.logspace(x_min, x_max, base=2, num=x_max-x_min+1)

# Configure the first subplot
axs[0].set_xlabel('Number of GPUs')
axs[0].set_ylabel('Timesteps/s')
axs[0].set_title('Strong Scaling Speed Timesteps/s')
axs[0].set_xscale('log', base=2)
axs[0].get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
axs[0].set_yscale('log')
axs[0].tick_params(axis='y', which='major', width=1, length=6)
axs[0].tick_params(axis='y', which='minor', width=1, length=3)
axs[0].grid(True)
axs[0].set_xticks(x_ticks)
axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Configure the second subplot
axs[1].set_xlabel('Number of GPUs')
axs[1].set_ylabel('Matom_step/s')
axs[1].set_title('Strong Scaling Throughput Matom_step/s')
axs[1].set_xscale('log', base=2)
axs[1].get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
axs[1].set_yscale('log')
axs[1].tick_params(axis='y', which='major', width=1, length=6)
axs[1].tick_params(axis='y', which='minor', width=1, length=3)
axs[1].grid(True)
axs[1].set_xticks(x_ticks)
axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.show()
plt.savefig("strong_scale.png")
