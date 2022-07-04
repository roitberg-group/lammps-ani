#ifndef ANI_LIB_H
#define ANI_LIB_H

#include <torch/script.h>
#include <torch/torch.h>
#include <cstdint>
#include <vector>

constexpr double hartree2kcalmol = 627.5094738898777;

class ANI {
public:
  torch::jit::script::Module model;
  torch::Device device;
  torch::Tensor atom_index12_t;
  torch::Tensor species_t;
  torch::Tensor species_ghost_as_padding_t;

  ANI() : device(torch::kCPU) {};
  ANI(const std::string& model_file, int local_rank);

  void compute(double& out_energy, std::vector<double>& out_force,
               std::vector<int64_t>& species, std::vector<double>& coordinates,
               int npairs_half, int64_t* atom_index12,
               int nlocal, int ago=0,
               std::vector<double>* out_atomic_energies=nullptr);
};

#endif
