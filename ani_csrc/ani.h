#ifndef ANI_LIB_H
#define ANI_LIB_H

#include <torch/script.h>
#include <torch/torch.h>
#include <cstdint>
#include <vector>

class ANI {
public:
  torch::jit::script::Module model;
  torch::Device device;

  ANI() : device(torch::kCPU) {};
  ANI(const std::string& model_file, int local_rank);

  void compute(double& out_energy, std::vector<float>& out_force,
               std::vector<int64_t>& species, std::vector<float>& coordinates,
               std::vector<int64_t>& atom_index12, std::vector<float>& diff_vector,
               std::vector<float>& distances, std::vector<int64_t>& ghost_index);
};

#endif