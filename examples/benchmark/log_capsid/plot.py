import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load the strong scaling data from the CSV file
df_44m = pd.read_csv('all.csv')

plt.figure(figsize=(4.5, 4), dpi=150)

# Plot Timesteps/s vs Number of GPUs for the 44M atom system
plt.plot(df_44m['num_gpus'], df_44m['timesteps/s'], 'o-', color='green', label='Capsid')

# Specify the range of your x-axis (modify as needed)
x_min = 0
x_max = int(np.log2(df_44m['num_gpus'].max()))
# Generate the ticks
x_ticks = np.logspace(x_min, x_max, base=2, num=x_max-x_min+1)

plt.xlabel('Number of GPUs')
plt.ylabel('Timesteps/s')
plt.title('44M Capsid Benchmark Speed Timesteps/s')
plt.ylim(1, 130)
plt.gca().set_xscale('log', base=2)
plt.gca().set_xticks(x_ticks)
plt.gca().set_yscale('log')
plt.gca().get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))

# Add a grid
plt.grid(True)

# Add a legend
plt.legend()

plt.show()

plt.savefig("capsid.png")
