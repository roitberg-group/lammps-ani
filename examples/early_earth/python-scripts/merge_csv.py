import glob
import pandas as pd

# Step 2: List all files that match the naming pattern
# file_paths = glob.glob('analyze/2023-10-13-163952*part*temp.csv')  # Adjust the pattern as needed
file_paths = ["analyze/2023-10-13-163952.474802.0.3.csv", "analyze/2023-10-13-163952.474802.1.3.csv", "analyze/2023-10-13-163952.474802.2.3.csv"]

print(len(file_paths))
print(file_paths)

# Step 3: Loop through the list of file paths and read each into a DataFrame
dfs = [pd.read_csv(file_path) for file_path in file_paths]

# Step 4: Concatenate all the individual DataFrames into a single DataFrame
merged_df = pd.concat(dfs, ignore_index=True)

merged_df.sort_values(by='frame', ascending=True, inplace=True)

# Optional: Save the merged DataFrame to a new CSV file
merged_df.to_csv('analyze/2023-10-13-163952.474802.all_fragments.all_frames.csv', index=False)

