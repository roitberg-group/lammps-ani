import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import matplotlib as mpl

# Set the font to Times New Roman
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 12

# Define a function to format the atoms per GPU
def format_atoms_per_gpu(x):
    return f'{x / 1e3:.1f}k'

# plt.style.use(['science','no-latex'])
# Function to generate intermediate colors
def generate_colors(start_color, end_color, n=5):
    def hex_to_rgb(value):
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i+lv//3], 16) for i in range(0, lv, lv//3))

    def rgb_to_hex(rgb):
        return '#%02x%02x%02x' % rgb
    n = n - 2
    # Convert the start and end colors to RGB
    start_color_rgb = hex_to_rgb(start_color)
    end_color_rgb = hex_to_rgb(end_color)

    # Generate the intermediate colors in RGB
    intermediate_colors_rgb = [tuple(np.linspace(start, end, n+2)[1:-1].astype(int)) for start, end in zip(start_color_rgb, end_color_rgb)]

    # Convert the intermediate colors to hex
    intermediate_colors_hex = [rgb_to_hex(color) for color in zip(*intermediate_colors_rgb)]

    return [start_color] + intermediate_colors_hex + [end_color]

# Define a list of 10 colors
colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
markers_ani = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h', 'd', '*']
markers_allegro = ['D', 'o', '^', 'D', 'v', 'P', 'X', 'h', 'd', '*']
# Generate intermediate colors for green and red
green_colors = generate_colors('#008564', '#88CC00', n=3)
red_colors = generate_colors('#D82727', '#DC5987', n=2)

# Read the csv file
df_ani = pd.read_csv('weak_ani.csv')
df_allegro = pd.read_csv('weak_allegro.csv')

# Calculate 'atoms_per_gpu' and add it as a new column to the dataframe
df_ani['atoms_per_gpu'] = df_ani['atoms'] / df_ani['num_gpus']
# Sort the dataframe by 'num_gpus' and 'atoms_per_gpu'
df_ani = df_ani.sort_values(['atoms_per_gpu', 'num_gpus'])
df_allegro = df_allegro.sort_values(['atoms_per_gpu', 'num_gpus'])

plt.figure(figsize=(6.5, 3.5), dpi=150)

plt.xlabel('Number of GPUs')
plt.plot([0, 0], [1, 1], marker=None, color=None, alpha=0, label=f'Atoms per GPU')

for i, atoms_per_gpu in enumerate(sorted(df_ani['atoms_per_gpu'].unique())):
    subset = df_ani[df_ani['atoms_per_gpu'] == atoms_per_gpu]
    plt.plot(subset['num_gpus'], subset['timesteps/s'], marker=markers_ani[i], color=green_colors[i], label=f'ANI      {format_atoms_per_gpu(atoms_per_gpu)}')

for i, atoms_per_gpu in enumerate(sorted(df_allegro['atoms_per_gpu'].unique())):
    subset = df_allegro[df_allegro['atoms_per_gpu'] == atoms_per_gpu]
    plt.plot(subset['num_gpus'], subset['timesteps/s'], marker=markers_allegro[i], color=red_colors[i], label=f'Allegro {format_atoms_per_gpu(atoms_per_gpu)}')

# Specify the range of your x-axis (modify as needed)
x_min = 0
x_max = max(int(np.log2(df_ani['num_gpus'].max())),
            int(np.log2(df_allegro['num_gpus'].max())))
# Generate the ticks
x_ticks = 2**np.arange(0, x_max+1, 2)  # Generate ticks at 1, 4, 16, 64, etc.

plt.ylabel('Speed (Timesteps/s)')
plt.gca().set_xscale('log', base=2)
plt.gca().get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
plt.title('Weak Scaling Benchmark (Water)')
plt.grid(True)
image_name = "weak_scale_log.png"
plt.gca().set_yscale('log', base=10)
plt.gca().set_xticks(x_ticks)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.subplots_adjust(right=0.65)

# Save the plot as a PNG file
plt.tight_layout()
plt.savefig(image_name)
