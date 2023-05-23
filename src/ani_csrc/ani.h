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
  torch::Dtype dtype;
  bool use_fullnbr;
  bool use_cuaev;
  bool use_single;

  torch::Tensor species_t;
  torch::Tensor species_ghost_as_padding_t;

  // for half nbr list
  torch::Tensor atom_index12_t;

  // for full nbr list
  torch::Tensor ilist_unique_t;
  torch::Tensor jlist_t;
  torch::Tensor numneigh_t;

  ANI() : device(torch::kCPU){};
  ANI(const std::string& model_file,
      int local_rank,
      int use_num_models = -1,
      bool use_cuaev_ = true,
      bool use_fullnbr_ = true,
      bool use_single_ = true);

  // compute with half nbrlist
  void compute(
      double& out_energy,
      std::vector<double>& out_force,
      std::vector<double>& out_virial,
      std::vector<int64_t>& species,
      std::vector<double>& coordinates,
      int npairs_half,
      int64_t* atom_index12,
      int nlocal,
      int ago = 0,
      std::vector<double>* out_atomic_energies = nullptr,
      bool vflag = false);

  // compute with full nbrlist
  void compute(
      double& out_energy,
      std::vector<double>& out_force,
      std::vector<double>& out_virial,
      std::vector<int64_t>& species,
      std::vector<double>& coordinates,
      int npairs,
      int* ilist_unique,
      int* jlist,
      int* numneigh,
      int nlocal,
      int ago = 0,
      std::vector<double>* out_atomic_energies = nullptr,
      bool vflag = false);

  // kokkos compute with full nbrlist
  void compute(
      torch::Tensor& out_energy,
      torch::Tensor& out_force,
      torch::Tensor& out_virial,
      torch::Tensor& species,
      torch::Tensor& coordinates,
      int npairs,
      torch::Tensor& ilist_unique,
      torch::Tensor& jlist,
      torch::Tensor& numneigh,
      int nlocal,
      int ago,
      torch::Tensor& out_atomic_energies,
      bool eflag_atom,
      bool vflag = false);
};

#endif
