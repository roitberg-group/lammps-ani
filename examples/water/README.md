Generate water box
```bash
python waterbox.py 6
```

Generate lammps data from pdb
```bash
python ../pdb2lmp.py water-6nm.pdb water-6nm.data --center
```

## Benchmark Result
Using `benchmark.sh`


| Kokkos | System Size (atom) | # of Models | # of GPUs | Performance (ns/day) |
|--------|--------------------|-------------|-----------|----------------------|
| yes    | 20k                | 1           | 1         | 6.003                |
| yes    | 20k                | 1           | 2         | 9.234                |
| yes    | 20k                | 1           | 4         | 11.267               |
| yes    | 20k                | 1           | 8         | 11.414               |
| yes    | 20k                | 8           | 1         | 2.128                |
| yes    | 20k                | 8           | 2         | 3.35                 |
| yes    | 20k                | 8           | 4         | 6.051                |
| yes    | 20k                | 8           | 8         | 8.238                |
| yes    | 300k               | 1           | 1         | 0.539                |
| yes    | 300k               | 1           | 2         | 1.04                 |
| yes    | 300k               | 1           | 4         | 1.469                |
| yes    | 300k               | 1           | 8         | 3.431                |
| yes    | 300k               | 8           | 1         | 0.162                |
| yes    | 300k               | 8           | 2         | 0.314                |
| yes    | 300k               | 8           | 4         | 0.603                |
| yes    | 300k               | 8           | 8         | 1.21                 |
| no     | 20k                | 1           | 1         | 2.874                |
| no     | 20k                | 1           | 2         | 5.069                |
| no     | 20k                | 1           | 4         | 8.106                |
| no     | 20k                | 1           | 8         | 11.086               |
| no     | 20k                | 8           | 1         | 1.502                |
| no     | 20k                | 8           | 2         | 2.816                |
| no     | 20k                | 8           | 4         | 4.636                |
| no     | 20k                | 8           | 8         | 8.077                |
| no     | 300k               | 1           | 1         | 0.193                |
| no     | 300k               | 1           | 2         | 0.373                |
| no     | 300k               | 1           | 4         | 0.749                |
| no     | 300k               | 1           | 8         | 1.451                |
| no     | 300k               | 8           | 1         | 0.106                |
| no     | 300k               | 8           | 2         | 0.209                |
| no     | 300k               | 8           | 4         | 0.421                |
| no     | 300k               | 8           | 8         | 0.813                |
