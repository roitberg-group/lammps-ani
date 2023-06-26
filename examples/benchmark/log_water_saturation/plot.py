import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load the strong scaling data from the CSV file
df = pd.read_csv('all.csv')

plt.figure(figsize=(5.5, 4), dpi=150)

# Plot Timesteps/s vs Number of GPUs for the 44M atom system
plt.plot(df['atoms'], df['Matom_step/s'], 'o-', color='green')

plt.xlabel('Number of Atoms')
plt.ylabel('Throughput (Matom_step/s)')
plt.title('Single GPU Throughput (Matom_step/s)')
formatter = ticker.FuncFormatter(lambda x, p: format(int(x), ','))
plt.gca().xaxis.set_major_formatter(formatter)


# Add a grid
plt.grid(True)

plt.show()

plt.savefig("saturation.png")
