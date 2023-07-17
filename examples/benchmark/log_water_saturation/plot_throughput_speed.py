import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load the strong scaling data from the CSV file
df = pd.read_csv('all.csv')

fig, axs = plt.subplots(2, 1, figsize=(5.5, 8), dpi=150)

# Plot Timesteps/s vs Number of GPUs for the 44M atom system
axs[0].plot(df['atoms'], df['Matom_step/s'], 'o-', color="blue")
axs[0].set_xlabel('Number of Atoms')
axs[0].set_ylabel('Throughput (Matom_step/s)')
axs[0].set_title('Single GPU Throughput (Matom_step/s)')
formatter = ticker.FuncFormatter(lambda x, p: format(int(x), ','))
axs[0].xaxis.set_major_formatter(formatter)
axs[0].xaxis.set_major_locator(ticker.MultipleLocator(base=200000))
axs[0].grid(True)

# Plot Timesteps/s vs Number of GPUs for the 44M atom system
axs[1].plot(df['atoms'], df['timesteps/s'], 'o-', color="blue")
axs[1].set_xlabel('Number of Atoms')
axs[1].set_ylabel('Speed (timesteps/s)')
axs[1].set_title('Single GPU Speed (timesteps/s)')
axs[1].xaxis.set_major_formatter(formatter)
axs[1].xaxis.set_major_locator(ticker.MultipleLocator(base=200000))
axs[1].grid(True)

plt.tight_layout()
plt.show()

fig.savefig("combined_saturation.png")

