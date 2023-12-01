# Importing required libraries
from ase import Atoms
import pandas as pd
import argparse

def rename_formula_in_csv(csv_path):
    # Reading the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Rename the existing 'formula' column to 'old_formula'
    df.rename(columns={'formula': 'old_formula'}, inplace=True)
   
    def convert_formula_using_ase(old_formula):
        atoms = Atoms(old_formula)
        formula = atoms.get_chemical_formula(mode='hill')
        return formula
    print("start") 
    # Apply the function to get new formula names
    df['formula'] = df['old_formula'].apply(convert_formula_using_ase)
    
    # Saving the updated DataFrame back to CSV
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rename formula in CSV file.')
    parser.add_argument('csv_path', type=str, help='Path to the CSV file')
    args = parser.parse_args()
    rename_formula_in_csv(args.csv_path)


