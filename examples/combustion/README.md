
run 300k system
```bash
python run_one.py prepare_system/combustion-0.25-300k.data --kokkos --num_gpus=8 --input_file=in.lammps --log_dir=logs --ani_model_file='ani2x_repulsion.pt' --run_name=combustion_300k --ani_num_models=1 --timestep=0.5 --run_steps=2000000 --run
```
