import argparse

def add_element_symbol(input_pdb, output_pdb):
    with open(input_pdb, 'r') as f_in, open(output_pdb, 'w') as f_out:
        for line in f_in:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom_name = line[12:16].strip()
                if atom_name == 'SOD' or atom_name == 'CLA':
                    continue  # Skip lines with atom name 'SOD' or 'CLA'
                element_symbol = atom_name[0] if atom_name else ' '  # First character of atom_name or empty space if atom_name is empty
                # Append extra spaces to make sure the length is at least 76
                if len(line) < 78:
                    line = line[:-1]  # remove \n
                    line += ' ' * (78 - len(line))
                    line += '\n'
                new_line = line[:76] + element_symbol + line[78:]  # Replace columns 77-78 with element_symbol
                f_out.write(new_line)
            else:
                f_out.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add element symbol to PDB file')
    parser.add_argument('input_pdb', help='Input PDB file')
    args = parser.parse_args()

    output_pdb = args.input_pdb.rsplit('.', 1)[0] + "-cleaned.pdb"
    add_element_symbol(args.input_pdb, output_pdb)

