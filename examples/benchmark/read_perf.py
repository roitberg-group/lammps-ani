import os
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
            num_gpus = int(re.search(r'on (\d+) procs', line).group(1))
            in_a_block = True
        elif 'Performance:' in line:
            assert in_a_block
            perf = [float(val) for val in re.findall(r'(\d+.\d+)', line)]
            perf_ns_day = perf[0]
            perf_timesteps_s = perf[2]
            perf_Matoms_step_s = perf[-1]
            data.append([steps, atoms, num_gpus, perf_ns_day, perf_timesteps_s, perf_Matoms_step_s])
            in_a_block = False

    # Convert the list to DataFrame
    df = pd.DataFrame(data, columns=['steps', 'atoms', 'num_gpus', 'ns/day', 'timesteps/s', 'Matom_step/s'])

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
    plt.close()


# Argument parser
parser = argparse.ArgumentParser(description='Process log files.')
parser.add_argument('path', type=str, help='The path to a directory or a log file')
parser.add_argument('--plot', action='store_true', help='If provided, also make a plot.')
args = parser.parse_args()

# Check if path is a file or a directory
if os.path.isfile(args.path):
    files_to_process = [args.path]  # Single file provided
elif os.path.isdir(args.path):
    # Get the list of all .log files in the directory
    files_to_process = sorted([os.path.join(args.path, f) for f in os.listdir(args.path) if f.endswith('.log')])

    # Check if there are .log files in the directory
    if not files_to_process:
        print("No .log files found in the directory.")
        exit()
else:
    print(f"Provided path {args.path} is not a valid file or directory.")
    exit()

# Process each log file
csv_files = []
for full_path in files_to_process:
    filename = os.path.basename(full_path)
    print(f"Processing {filename}...")

    # Call the function and print the dataframe
    df = extract_data_from_log(full_path)
    print(df.iloc[-1:])

    log_path = Path(full_path)
    csv_file = log_path.parent / f"{log_path.stem}.csv"
    df.to_csv(csv_file, index=False)
    print(f"Csv saved to {csv_file}")
    csv_files.append(csv_file)

    if args.plot:
        plot(df)

print("Finished processing all files.")

# Readall csv files and extract the last row from each csv and merge to a single csv
data = []
# Iterate over all the CSV files
for csv_file in csv_files:
    # Read the CSV file
    df = pd.read_csv(csv_file)
    # Get the last row
    last_row = df.iloc[-1]
    # Add the last row to the data list
    data.append(last_row)

# Convert the list into a DataFrame
df_final = pd.DataFrame(data).reset_index(drop=True)
df_final.sort_values(by=['atoms', 'num_gpus'], inplace=True)
df_final.drop('steps', axis=1, inplace=True)
final_csv_file = log_path.parent / f"all.csv"
df_final.to_csv(final_csv_file, index=False)

print(f"Saved all data to {final_csv_file}")
print(df_final)
