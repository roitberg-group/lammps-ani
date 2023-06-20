Hi, I would like you assist me on writing a readme document on how I conducted the benchmark.

It including a few sections
### data preparation

#### Water box sysetem
Generate pdb file using `./data/water/prepare/generate_pdb.py`. This script will generate pdb files for water system range from 50k atoms to 10M atoms. Internally, it calculate the density and use packmol to build the system. The 10M atoms system takes 2 hour to generate. The 100M atoms system takes 2 days to generate.

Then convert pdb to lammps data, `pdb2lmp.py` could be found at [../pdb2lmp.py](../pdb2lmp.py).
```bash
python path/to/pdb2lmp.py water.pdb water.data
```

Finally, move all lammps data files from `data/water/prepare` to `data/water` folder

#### HIV Capsid system

We got the HIV capsid structures from Gregory Voth.
- `data/capsid-aa/capsid-cone.pdb` consists of 70M atoms, with 65.6M of them are water.
- `data/capsid-aa/capsid5/capsid-pill.pdb` consists of 44M atoms, with 40M of them are water. This is the system we used for benchmark.

We need to do some pre-processing for the pdb files using script at `data/capsid-aa/capsid5/clean_pdb.py`. It does two things:
1. Removes SOD and CLA residue.
2. Add element symbol to each atom in the pdb so that ASE could read the pdb correctly

Then as usual, convert the pdb to lammps data using ``pdb2lmp.py`. The final lammps data is at `data/capsid-aa/capsid5/capsid-pill-cleaned.data`.

### relax the water system
Run lammps with NVT for 5k steps, so the system is relaxed, the temperature is stabilized at 300K.
For example:
```
python run_one.py data/water/water-50k.data --kokkos --num_gpus=4 --run_steps=5000 --input_file=in.relax.lammps --log_dir=log_water_relax --run
```
You could adjust the number of gpus if you need to run for very large system e.g. 10M system. For small system, each job normally finish within a few minutes. When the relaxition is done, the final state is saved to `data/water/water-50k.data.final` and is ready to use and benchmark.

### weak scaling
Weak scaling benchmarks are used to evaluate the ability of a system to maintain its performance as the size of the problem increases proportionally to the resources. In other words, double the size of the problem and double the amount of resources, perf should stay the same.

We conducted weak scaling for water system for 4 workloads size per gpu (50k, 100k, 200k, 400k). The number of gpus is increased from 1 to 48.
We used kokkos, single precision, FP32, timestep 0.5, 1 single model. The benchmark is conducted on hipergator with 8 A100 GPUs per node.

System scaling is achieved by `replicate` command in lammps.

To run the benchmark
```bash
python submit_scaling.py data/water-200k.data.final
```

The benchmark result:

| atoms_per_gpu | atoms    | num_gpus | ns/day | timesteps/s | Matoms_step/s |
|---------------|----------|----------|--------|-------------|---------------|
| 50001         | 50001    | 1        | 2.826  | 65.41       | 3.271         |
| 50001         | 100002   | 2        | 2.782  | 64.392      | 6.439         |
| 50001         | 200004   | 4        | 2.758  | 63.839      | 12.768        |
| 50001         | 400008   | 8        | 2.701  | 62.526      | 25.011        |
| 50001         | 800016   | 16       | 2.546  | 58.924      | 47.14         |
| 50001         | 1600032  | 32       | 2.474  | 57.264      | 91.624        |
| 50001         | 2400048  | 48       | 2.429  | 56.224      | 134.939       |
| 100002        | 100002   | 1        | 1.492  | 34.542      | 3.454         |
| 100002        | 200004   | 2        | 1.481  | 34.276      | 6.855         |
| 100002        | 400008   | 4        | 1.473  | 34.089      | 13.636        |
| 100002        | 800016   | 8        | 1.457  | 33.731      | 26.985        |
| 100002        | 1600032  | 16       | 1.405  | 32.529      | 52.047        |
| 100002        | 3200064  | 32       | 1.365  | 31.593      | 101.1         |
| 100002        | 4800096  | 48       | 1.346  | 31.153      | 149.538       |
| 200001        | 200001   | 1        | 0.772  | 17.871      | 3.574         |
| 200001        | 400002   | 2        | 0.769  | 17.8        | 7.12          |
| 200001        | 800004   | 4        | 0.766  | 17.733      | 14.187        |
| 200001        | 1600008  | 8        | 0.761  | 17.622      | 28.196        |
| 200001        | 3200016  | 16       | 0.739  | 17.111      | 54.757        |
| 200001        | 6400032  | 32       | 0.723  | 16.734      | 107.101       |
| 200001        | 9600048  | 48       | 0.718  | 16.625      | 159.602       |
| 400002        | 400002   | 1        | 0.394  | 9.112       | 3.645         |
| 400002        | 800004   | 2        | 0.393  | 9.097       | 7.278         |
| 400002        | 1600008  | 4        | 0.392  | 9.077       | 14.523        |
| 400002        | 3200016  | 8        | 0.39   | 9.029       | 28.894        |
| 400002        | 6400032  | 16       | 0.382  | 8.849       | 56.636        |
| 400002        | 12800064 | 32       | 0.376  | 8.7         | 111.361       |
| 400002        | 19200096 | 48       | 0.374  | 8.662       | 166.318       |

Here is the plot

![](resc/weak_scale.png)

<!-- Explain the weak scaling plot *briefly* -->


<!-- compare to allegro -->
For a comparison, we run 400k atoms per GPU, and are able to reach to about 9 timesteps/second. We used FP32.
Allegro run 12.5k atoms perg GPU, and the performance is about 8 timesteps/s. At the meanwhile, allegro uses TF32 that could has a much higher performance on matrix multiplications.

### strong scaling
Strong scaling, also known as scale-up, is a concept in parallel computing that measures the performance improvement of a system as more resources (like processors or GPUs) are added, while keeping the total problem size or workload constant. In other words, solve a fixed-size problem faster with more GPUs.

We tested on 3 systems 300k, 1M and 10M atoms system. The number of gpus is increased from 1 to 56.

To run the benchmark
```bash
python submit_scaling.py data/water-200k.data.final
```

| atoms    | num_gpus | ns/day | timesteps/s | Matoms_step/s |
|----------|----------|--------|-------------|---------------|
| 300003   | 1        | 0.521  | 12.065      | 3.62          |
| 300003   | 2        | 1.005  | 23.259      | 6.978         |
| 300003   | 4        | 1.914  | 44.305      | 13.292        |
| 300003   | 8        | 3.446  | 79.761      | 23.928        |
| 300003   | 16       | 5.371  | 124.332     | 37.3          |
| 300003   | 32       | 7.678  | 177.728     | 53.319        |
| 300003   | 48       | 8.76   | 202.769     | 60.831        |
| 300003   | 56       | 8.839  | 204.61      | 61.384        |
| 1000002  | 1        | 0.159  | 3.684       | 3.684         |
| 1000002  | 2        | 0.315  | 7.287       | 7.287         |
| 1000002  | 4        | 0.615  | 14.24       | 14.24         |
| 1000002  | 8        | 1.182  | 27.354      | 27.354        |
| 1000002  | 16       | 2.118  | 49.038      | 49.038        |
| 1000002  | 32       | 3.659  | 84.695      | 84.695        |
| 1000002  | 48       | 4.823  | 111.634     | 111.634       |
| 1000002  | 56       | 5.116  | 118.424     | 118.424       |
| 10000002 | 8        | 0.126  | 2.927       | 29.267        |
| 10000002 | 16       | 0.244  | 5.646       | 56.461        |
| 10000002 | 32       | 0.474  | 10.976      | 109.76        |
| 10000002 | 48       | 0.693  | 16.035      | 160.35        |
| 10000002 | 56       | 0.797  | 18.443      | 184.431       |


Here is the plot

![](resc/strong_scale.png)

<!-- Explain the weak scaling plot *briefly* -->


<!-- compare to allegro -->
For a comparison,
ANI, 1M atoms (water), 48 GPUs, 112 timesteps/sec, (we are using FP32) 
Allegro, 1M atoms(water), 2048 GPUs, 100 timesteps/sec, (they are using TF32!)

### 44M biosystem of capsid

To run the benchmark
```bash
python submit_scaling.py data/capsid-aa/capsid5/capsid-pill-cleaned.data
```

| atoms    | num_gpus | ns/day | timesteps/s | Matoms_step/s |
|----------|----------|--------|-------------|---------------|
| 43911876 | 24       | 0.072  | 1.667       | 73.215        |
| 43911876 | 24       | 0.072  | 1.666       | 73.148        |
| 43911876 | 32       | 0.098  | 2.276       | 99.946        |
| 43911876 | 32       | 0.098  | 2.271       | 99.712        |
| 43911876 | 40       | 0.121  | 2.807       | 123.262       |
| 43911876 | 48       | 0.146  | 3.37        | 147.964       |
| 43911876 | 56       | 0.17   | 3.936       | 172.835       |
| 43911876 | 64       | 0.192  | 4.445       | 195.173       |

Here is the plot

![44M](resc/capsid.png)

For a comparison:
Our speed:
3.9 timesteps/s on 56 GPUs. (FP32)
Allegro:
3.9 timesteps/s on 2048 GPUs
8.7 timesteps/s on 5120 GPUs

### saturation test on a single GPU
The goal of this benchmarking is to determine the saturation point of a single GPU, beyond which increasing the system size does not yield further performance improvement.

| atoms   | ns/day | timesteps/s | Matoms_step/s |
|---------|--------|-------------|---------------|
| 50001   | 2.827  | 65.446      | 3.272         |
| 100002  | 1.495  | 34.617      | 3.462         |
| 200001  | 0.772  | 17.869      | 3.574         |
| 300003  | 0.521  | 12.065      | 3.62          |
| 400002  | 0.394  | 9.122       | 3.649         |
| 500001  | 0.316  | 7.315       | 3.658         |
| 600000  | 0.265  | 6.133       | 3.68          |
| 700002  | 0.227  | 5.257       | 3.68          |
| 800001  | 0.199  | 4.608       | 3.686         |
| 900000  | 0.177  | 4.1         | 3.69          |

We could see that the performance is saturated around 3.68 Matoms_step/s at 500k.
We believe this could be improved by reducing memory reallocation, CPU syncronization, etc.
