#ifndef ANI_LIB_H
#define ANI_LIB_H

#include <torch/script.h>
#include <torch/torch.h>
#include <cstdint>
#include <vector>

constexpr float hartree2kcalmol = 627.5094738898777;

class ANI {
public:
  torch::jit::script::Module model;
  torch::Device device;
  torch::Tensor species_t;
  torch::Tensor species_ghost_as_padding_t;

  // for half nbr list
  torch::Tensor atom_index12_t;

  // for full nbr list
  torch::Tensor ilist_unique_t;
  torch::Tensor numneigh_t;
  torch::Tensor jlist_t;

  ANI() : device(torch::kCPU) {};
  ANI(const std::string& model_file, int local_rank);

  // compute with half nbrlist
  void compute(double& out_energy, std::vector<float>& out_force,
               std::vector<int64_t>& species, std::vector<float>& coordinates,
               int npairs_half, int64_t* atom_index12,
               int nlocal, int ago=0);
  // compute with full nbrlist
  void compute(double& out_energy, std::vector<float>& out_force,
                  std::vector<int64_t>& species, std::vector<float>& coordinates,
                  int* ilist_unique, int* numneigh, int nlocal,
                  int* jlist, int npairs,
                  int ago=0);
};

#endif
