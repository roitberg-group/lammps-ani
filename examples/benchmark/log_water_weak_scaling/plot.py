# Complete code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.ticker as ticker


# plt.style.use(['science','no-latex'])

# Read the csv file
df = pd.read_csv('all.csv')

# Calculate 'atoms_per_gpu' and add it as a new column to the dataframe
df['atoms_per_gpu'] = df['atoms'] / df['num_gpus']

# Sort the dataframe by 'num_gpus' and 'atoms_per_gpu'
df = df.sort_values(['atoms_per_gpu', 'num_gpus'])
# df.to_csv("all.csv", index=False)

# Define a function to format the atoms per GPU
def format_atoms_per_gpu(x):
    return f'{x / 1e3:.0f}k'

colors = ['purple', 'red', 'blue', 'green', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
markers = ['D', 'o', 's', 'v', '^', 'P', 'X', 'h', 'd', '*']

plt.figure(figsize=(6, 4), dpi=150)

for i, atoms_per_gpu in enumerate(sorted(df['atoms_per_gpu'].unique())):
    subset = df[df['atoms_per_gpu'] == atoms_per_gpu]
    plt.plot(subset['num_gpus'], subset['timesteps/s'], marker=markers[i], color=colors[i], label=f'Atoms per GPU: {format_atoms_per_gpu(atoms_per_gpu)}')

# Specify the range of your x-axis (modify as needed)
x_min = 0
x_max = int(np.log2(df['num_gpus'].max()))
# Generate the ticks
x_ticks = np.logspace(x_min, x_max, base=2, num=x_max-x_min+1)

plt.xlabel('Number of GPUs')
plt.ylabel('Speed (Timesteps/s)')
plt.gca().set_xscale('log', base=2)
plt.gca().get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
plt.title('Water Weak Scaling')
plt.grid(True)
plt.gca().set_xticks(x_ticks)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.subplots_adjust(right=0.65)

# Save the plot as a PNG file
plt.savefig('weak_scale.png')

