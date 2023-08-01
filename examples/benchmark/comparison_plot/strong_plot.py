import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import matplotlib as mpl

# Set the font to Times New Roman
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 12


def human_format(num):
    # Function to format numbers to human readable format
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.0f%s' % (num, ['', 'K', 'M', 'B', 'T', 'P'][magnitude])

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
markers = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h', 'd', '*']
# Generate intermediate colors for green and red
green_colors = generate_colors('#008564', '#88CC00', n=4)
red_colors = generate_colors('#D82727', '#DC5987', n=4)

fig, axs = plt.subplots(2, 1, figsize=(7, 6), dpi=150, sharex=True)

x_label = 'Number of GPUs'
filename = "strong_scale.png"

# ################################ plot for bio system #########################################

# Load the strong scaling data from the CSV file
df_strong_scaling_ani = pd.read_csv('strong_ani_capsid.csv')  # update this with the path to your CSV file
df_strong_scaling_allegro = pd.read_csv('strong_allegro_capsid.csv')  # update this with the path to your CSV file

# Get all unique atom counts
atom_counts = df_strong_scaling_ani['atoms'].unique()
atom_counts_allegro = df_strong_scaling_allegro['atoms'].unique()

# Raise an error if there are more than 10 unique atom counts
if len(atom_counts) > 10:
    raise ValueError("Number of unique atom counts exceeds 10")

# Iterate over the unique atom counts
for i, atom_count in enumerate(atom_counts):
    # Filter the data for the current atom count
    df_atom = df_strong_scaling_ani[df_strong_scaling_ani['atoms'] == atom_count]

    # Plot the Timesteps/s vs Number of GPUs
    axs[0].plot(df_atom['num_gpus'], df_atom['timesteps/s'], marker=markers[i], color=green_colors[i], label=f'ANI      {human_format(atom_count)} atoms')

# Add this code after the loop that plots the Ani data:
for i, atom_count in enumerate(atom_counts_allegro):
    # Filter the data for the current atom count
    df_atom_allegro = df_strong_scaling_allegro[df_strong_scaling_allegro['atoms'] == atom_count]

    # Plot the Timesteps/s vs Number of GPUs
    axs[0].plot(df_atom_allegro['num_gpus'], df_atom_allegro['timesteps/s'], marker=markers[i], color=red_colors[i], label=f'Allegro {human_format(atom_count)} atoms')

# ################################ plot for water system #########################################

# Load the strong scaling data from the CSV file
df_strong_scaling_ani = pd.read_csv('strong_ani.csv')  # update this with the path to your CSV file
df_strong_scaling_allegro = pd.read_csv('strong_allegro.csv')  # update this with the path to your CSV file

# Get all unique atom counts
atom_counts = df_strong_scaling_ani['atoms'].unique()
atom_counts_allegro = df_strong_scaling_allegro['atoms'].unique()

# Raise an error if there are more than 10 unique atom counts
if len(atom_counts) > 10:
    raise ValueError("Number of unique atom counts exceeds 10")

# Iterate over the unique atom counts
for i, atom_count in enumerate(atom_counts):
    # Filter the data for the current atom count
    df_atom = df_strong_scaling_ani[df_strong_scaling_ani['atoms'] == atom_count]

    # Plot the Timesteps/s vs Number of GPUs
    axs[1].plot(df_atom['num_gpus'], df_atom['timesteps/s'], marker=markers[i], color=green_colors[i], label=f'ANI      {human_format(atom_count)} atoms')

# Add this code after the loop that plots the Ani data:
for i, atom_count in enumerate(atom_counts_allegro):
    # Filter the data for the current atom count
    df_atom_allegro = df_strong_scaling_allegro[df_strong_scaling_allegro['atoms'] == atom_count]

    # Plot the Timesteps/s vs Number of GPUs
    axs[1].plot(df_atom_allegro['num_gpus'], df_atom_allegro['timesteps/s'], marker=markers[i], color=red_colors[i], label=f'Allegro {human_format(atom_count)} atoms')

# Specify the range of your x-axis (modify as needed)
x_min = 0
x_max = max(int(np.log2(df_strong_scaling_ani['num_gpus'].max())),
            int(np.log2(df_strong_scaling_allegro['num_gpus'].max())))
# Generate the ticks
x_ticks = 2**np.arange(0, x_max+1, 2)  # Generate ticks at 1, 4, 16, 64, etc.

# if it is an interger return interger else return float
def format_func(value, tick_number):
    return int(value) if value.is_integer() else value

# Configure the first subplot
axs[0].set_ylabel('Speed (Timesteps/s)')
axs[0].set_title('Strong Scaling Benchmark\nHIV Capsid')
axs[0].set_xscale('log', base=2)
axs[0].get_xaxis().set_major_formatter(ticker.FuncFormatter(format_func))
axs[0].set_yscale('log')
axs[0].tick_params(axis='y', which='major', width=1, length=6)
axs[0].tick_params(axis='y', which='minor', width=1, length=3)
axs[0].grid(True)
axs[0].set_ylim(1, 300)
axs[0].set_xticks(x_ticks)
axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Configure the first subplot
axs[1].set_xlabel(x_label)
axs[1].set_ylabel('Speed (Timesteps/s)')
axs[1].set_title('Water')
axs[1].set_xscale('log', base=2)
axs[1].get_xaxis().set_major_formatter(ticker.FuncFormatter(format_func))
axs[1].set_yscale('log')
axs[1].tick_params(axis='y', which='major', width=1, length=6)
axs[1].tick_params(axis='y', which='minor', width=1, length=3)
axs[1].grid(True)
axs[1].set_ylim(1, 300)
axs[1].set_xticks(x_ticks)
axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
plt.tight_layout()
plt.savefig(filename)
