import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse

# Load the strong scaling data from the CSV file
df_44m = pd.read_csv('all.csv')

# gpus_per_node variable
gpus_per_node = 4

# Argument parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--node", help="plot the x-axis in number of nodes instead of number of gpus", action="store_true")
args = parser.parse_args()

plt.figure(figsize=(4, 3), dpi=150)

# Check if the node flag is passed
if args.node:
    df_44m['num_gpus'] = df_44m['num_gpus'] / gpus_per_node
    plt.xlabel('Number of Nodes')
    image_name = "capsid_nodes.png"
else:
    plt.xlabel('Number of GPUs')
    image_name = "capsid.png"

# Plot Timesteps/s vs Number of GPUs for the 44M atom system
plt.plot(df_44m['num_gpus'], df_44m['timesteps/s'], 'o-', color='green', label='Capsid')

# Specify the range of your x-axis (modify as needed)
x_min = 0
if args.node:
    x_max = 10
    # Generate the ticks
    x_ticks = np.logspace(x_min, x_max, base=2, num=int((x_max-x_min)/2)+1)
    plt.xlim(0.9,)
else:
    x_max = int(np.log2(df_44m['num_gpus'].max()))
    # Generate the ticks
    x_ticks = np.logspace(x_min, x_max, base=2, num=x_max-x_min+1)

plt.ylabel('Speed (Timesteps/s)')
plt.title('44M Capsid Benchmark')
plt.ylim(1, 130)
plt.gca().set_xscale('log', base=2)
plt.gca().set_xticks(x_ticks)
plt.gca().set_yscale('log')
plt.gca().get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))

# Add a grid
plt.grid(True)

# Add a legend
plt.legend()
plt.tight_layout()
plt.show()

plt.savefig(image_name)
