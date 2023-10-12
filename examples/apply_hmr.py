import argparse
import openmm.app as omm_app
from openmm import unit

def apply_hmr(input_pdb, output_mass_file, hydrogen_mass_value):
    print(f"Reading PDB from: {input_pdb}")
    
    hydrogenMass = hydrogen_mass_value * unit.amu
    pdb = omm_app.PDBFile(input_pdb)

    forcefield = omm_app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=omm_app.NoCutoff, hydrogenMass=hydrogenMass, rigidWater=False)

    print("\nFirst 20 atoms - Masses before and after HMR:")

    # Combine processing and writing to file
    with open(output_mass_file, "w") as outfile:
        outfile.write("Hmrmass\n\n")
        
        for atom, index in zip(pdb.topology.atoms(), range(system.getNumParticles())):
            mass_before = atom.element.mass.value_in_unit(unit.dalton)
            mass_hmr = system.getParticleMass(index).value_in_unit(unit.dalton)

            # Print the first 20 atoms' information
            if index < 20:
                print(f"Index: {index + 1:02}, Element: {atom.element.symbol}, Mass before: {mass_before:7.4f}, after HMR: {mass_hmr:7.4f}")

            outfile.write(f"{index + 1} {mass_hmr}\n")

    print(f"\nThe output is saved at: {output_mass_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Hydrogen Mass Repartitioning (HMR) to a PDB and output atomic masses.")
    parser.add_argument("input_pdb", help="Path to the input PDB file.")
    parser.add_argument("output_mass_file", help="Path to the output atomic mass file.")
    parser.add_argument("--hydrogenMass", type=float, default=3, help="Desired mass for hydrogen atoms. Default is 3 amu.")

    args = parser.parse_args()

    apply_hmr(args.input_pdb, args.output_mass_file, args.hydrogenMass)

