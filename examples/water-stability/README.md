# Water Stability Test - TIP3P

## Requirements

**Run 1: Rigid water (SHAKE constraints)**
- System: 216 water molecules (648 atoms)
- Force field: Classical TIP3P
- Phase 1: NPT equilibration, 5 ns, 300 K, 1 atm
- Phase 2: NVT production, 10 ns, 300 K
- Timestep: 1.0 fs
- Constraints: SHAKE on O-H bonds
- Data file: `water-216.data` (with bond information)
- Output: Trajectory every 100 steps → `.lammpstrj`

**Run 2: Flexible water (no constraints)**
- Same system and phases as Run 1
- Timestep: 0.2 fs (smaller due to O-H bond vibrations)
- NO SHAKE constraints
- Data file: `water-216.data` (same file as Run 1)


## Run

```bash
sbatch submit_run1.sh   # ~4 hours
sbatch submit_run2.sh   # ~12 hours
```

## Files

- `tip3p.mol` - Water molecule template
- `in.run1.lammps` - SHAKE input
- `in.run2.lammps` - Flexible input
- `run1/`, `run2/` - Output directories (traj.dcd, output.log)
