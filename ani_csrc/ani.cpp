#include "ani.h"
#include <torch/script.h>
#include <torch/torch.h>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <vector>

ANI::ANI(const std::string& model_file, int local_rank) : device(local_rank == -1 ? torch::kCPU : torch::kCUDA, local_rank) {
  at::globalContext().setAllowTF32CuBLAS(false);
  at::globalContext().setAllowTF32CuDNN(false);
  try {
    model = torch::jit::load(model_file, device);
    std::cout << "Successfully loaded the model on " << device << std::endl;
  } catch (const c10::Error& e) {
    std::cerr << "Error loading the model on " << device << std::endl;
  }
}

void ANI::compute(
    double& out_energy,
    std::vector<double>& out_force,
    std::vector<int64_t>& species,
    std::vector<double>& coordinates,
    int npairs_half,
    int64_t* atom_index12,
    int nlocal,
    int ago,
    std::vector<double>* out_atomic_energies) {
  int ntotal = species.size();

  // output tensor
  auto out_force_t = torch::from_blob(out_force.data(), {1, ntotal, 3}, torch::dtype(torch::kFloat64));
  // input tensor
  auto coordinates_t =
      torch::from_blob(coordinates.data(), {1, ntotal, 3}, torch::dtype(torch::kFloat64)).to(device).requires_grad_(true);

  // species_t and atom_index12_t are cloned/cached on devices and only needs to be updated when neigh_list rebuild
  if (ago == 0) {
    atom_index12_t = torch::from_blob(atom_index12, {2, npairs_half}, torch::dtype(torch::kLong)).to(device);
    species_t = torch::from_blob(species.data(), {1, ntotal}, torch::dtype(torch::kLong)).to(device);
    // when runing on the CPU, we have to explicitly clone these two tensors
    // because they are created from temporary vector data pointers
    if (device == torch::kCPU) {
      atom_index12_t = atom_index12_t.clone();
      species_t = species_t.clone();
    }
    species_ghost_as_padding_t = species_t.detach().clone();
    // equivalent to: species_ghost_as_padding[:, nlocal:] = -1
    species_ghost_as_padding_t.index_put_({torch::indexing::Slice(), torch::indexing::Slice(nlocal, torch::indexing::None)}, -1);
  }

  // perform calculation of diff and dist on device
  auto diff_vector_t = coordinates_t.squeeze(0).index({atom_index12_t[0]}) - coordinates_t.squeeze(0).index({atom_index12_t[1]});
  auto distances_t = diff_vector_t.norm(2, -1);

  // pack forward inputs
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(species_t);
  inputs.push_back(coordinates_t);
  inputs.push_back(atom_index12_t);
  inputs.push_back(diff_vector_t);
  inputs.push_back(distances_t);
  inputs.push_back(species_ghost_as_padding_t);
  bool atomic = out_atomic_energies != nullptr;
  inputs.push_back(atomic);

  // run ani model
  torch::Tensor energy, force, atomic_energies;
  auto outputs = model.forward(inputs).toTuple();
  // extract energy and force from model outputs, and convert the unit to kcal/mol
  energy = outputs->elements()[0].toTensor() * hartree2kcalmol;
  force = outputs->elements()[1].toTensor() * hartree2kcalmol;

  // write energy and force out
  out_energy = energy.item<double>();
  out_force_t.copy_(force);

  // if atomic is false, atomic_energies will be an empty tensor
  if (atomic) {
    atomic_energies = outputs->elements()[2].toTensor() * hartree2kcalmol;
    auto out_atomic_energies_t = torch::from_blob(out_atomic_energies->data(), {1, nlocal}, torch::dtype(torch::kFloat64));
    out_atomic_energies_t.copy_(atomic_energies);
  }
}
