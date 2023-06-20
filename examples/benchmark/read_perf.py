import re
import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


# Function to extract data from log
def extract_data_from_log(filename):
    # Open the file
    with open(filename, 'r') as file:
        # Read the lines from the file
        lines = file.readlines()

    # Prepare an empty list to hold the data
    data = []

    # Iterate over the lines
    for line in lines:
        if 'steps with' in line:
            steps = int(re.search(r'(\d+) steps', line).group(1))
            atoms = int(re.search(r'with (\d+) atoms', line).group(1))
            in_a_block = True
        elif 'Performance:' in line:
            assert in_a_block
            perf = [float(val) for val in re.findall(r'(\d+.\d+)', line)]
            perf_ns_day = perf[0]
            perf_timesteps_s = perf[2]
            perf_Matoms_step_s = perf[-1]
            data.append([steps, atoms, perf_ns_day, perf_timesteps_s, perf_Matoms_step_s])
            in_a_block = False

    # Convert the list to DataFrame
    df = pd.DataFrame(data, columns=['steps', 'atoms', 'ns/day', 'timesteps/s', 'Matoms_step/s'])

    return df


def plot(df):
    fig, axs = plt.subplots(3, figsize=(7, 9), dpi=150)

    # Accumulate steps
    df['accumulated_steps'] = df['steps'].cumsum()

    # Plot 'ns/day' vs 'accumulated_steps'
    axs[0].plot(df['accumulated_steps'], df['ns/day'])
    axs[0].set_xlabel('Accumulated Steps')
    axs[0].set_ylabel('ns/day')
    axs[0].set_title('ns/day vs Accumulated Steps')

    # Plot 'timesteps/s' vs 'accumulated_steps'
    axs[1].plot(df['accumulated_steps'], df['timesteps/s'])
    axs[1].set_xlabel('Accumulated Steps')
    axs[1].set_ylabel('timesteps/s')
    axs[1].set_title('timesteps/s vs Accumulated Steps')

    # Plot 'Matoms_step/s' vs 'accumulated_steps'
    axs[2].plot(df['accumulated_steps'], df['Matoms_step/s'])
    axs[2].set_xlabel('Accumulated Steps')
    axs[2].set_ylabel('Matoms_step/s')
    axs[2].set_title('Matoms_step/s vs Accumulated Steps')

    plt.tight_layout()
    # plt.show()
    png_file = log_path.parent / f"{log_path.stem}.png"
    plt.savefig(png_file)

    print(f"png saved to {png_file}")


# Argument parser
parser = argparse.ArgumentParser(description='Process a log file.')
parser.add_argument('filename', type=str, help='The filename of the log file to process')
parser.add_argument('--plot', action='store_true', help='If provided, also make a plot.')
args = parser.parse_args()

# Call the function and print the dataframe
df = extract_data_from_log(args.filename)
print(df)

log_path = Path(args.filename)
csv_file = log_path.parent / f"{log_path.stem}.csv"
df.to_csv(csv_file, index=False)
print(f"csv saved to {csv_file}")

if args.plot:
    plot(df)
