Generate water box
```bash
python waterbox.py 6
```

Generate lammps data from pdb
```bash
python ../pdb2lmp.py water-6nm.pdb water-6nm.data --center
```


Benchmark Result


20k system
| # GPU | 1 model, no kokkos (ns/day) | 1 model, with kokkos (ns/day) | 8 model, no kokkos (ns/day) | 8 model, with kokkos (ns/day) |
|-------|-----------------------------|-------------------------------|-----------------------------|-------------------------------|
| 1     | 2.862                       | 5.967                         | 1.500                       | 2.116                         |
| 8     | 11.177                      | 11.396                        | 8.106                       | 8.299                         |

300k system
| # GPU | 1 model, no kokkos (ns/day) | 1 model, with kokkos (ns/day) | 8 model, no kokkos (ns/day) | 8 model, with kokkos (ns/day) |
|-------|-----------------------------|-------------------------------|-----------------------------|-------------------------------|
| 1     | 0.189                       | 0.536                         | 0.103                       | 0.161                         |
| 8     | 1.414                       | 3.424                         | 0.815                       | 1.204                         |
