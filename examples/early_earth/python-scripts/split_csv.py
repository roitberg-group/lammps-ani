import argparse
import pandas as pd
from pathlib import Path

def split_csv(file_path, num_parts):
    # Read the entire CSV file
    df = pd.read_csv(file_path)
    
    # Calculate the number of rows in each part
    rows_per_part = len(df) // num_parts
    
    # Use pathlib to manipulate file paths
    file_path_obj = Path(file_path)
    base_name = file_path_obj.stem
    extension = file_path_obj.suffix
    parent_dir = file_path_obj.parent
    
    # Loop through each part to save it as a new CSV file
    for i in range(num_parts):
        start_index = i * rows_per_part
        end_index = (i + 1) * rows_per_part if i != num_parts - 1 else len(df)
        part_df = df.iloc[start_index:end_index]
        
        # Create the name for the new CSV file
        new_file_name = f"{base_name}_part_{i+1}_of_{num_parts}{extension}"
        new_file_path = parent_dir / new_file_name
        
        # Save the DataFrame as a new CSV file
        part_df.to_csv(new_file_path, index=False)
        print(f"Saved part {i+1} as {new_file_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a CSV file into multiple parts.")
    parser.add_argument("file_path", type=str, help="Path to the CSV file to split.")
    parser.add_argument("num_parts", type=int, help="Number of parts to split the CSV file into.")
    
    args = parser.parse_args()
    split_csv(args.file_path, args.num_parts)
