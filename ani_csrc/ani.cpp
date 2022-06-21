#include <torch/script.h>
#include <torch/torch.h>
#include <tuple>
#include <vector>
#include <iostream>
#include <cstdint>
#include "ani.h"

ANI::ANI(const std::string& model_file, int local_rank) : device(local_rank == -1 ? torch::kCPU: torch::kCUDA, local_rank) {
  at::globalContext().setAllowTF32CuBLAS(false);
  at::globalContext().setAllowTF32CuDNN(false);
  try {
    model = torch::jit::load(model_file, device);
    std::cout << "Successfully loaded the model on " << device << std::endl;
  }
  catch (const c10::Error &e) {
    std::cerr << "Error loading the model on " << device << std::endl;
  }
}

// For simplicity, the accumulated energy will be saved into eng_vdwl,
// instead of writing to per atom energy.
void ANI::compute(double& out_energy, std::vector<double>& out_force,
                  std::vector<int64_t>& species, std::vector<double>& coordinates,
                  int npairs_half, int64_t* atom_index12,
                  int nlocal, int ago) {
  int ntotal = species.size();

  // output tensor
  auto out_force_t = torch::from_blob(out_force.data(), {1, ntotal, 3}, torch::dtype(torch::kFloat64));
  // input tensor
  auto coordinates_t = torch::from_blob(coordinates.data(), {1, ntotal, 3}, torch::dtype(torch::kFloat64)).to(device).requires_grad_(true);

  // atom_index12_t is cached on GPU and only needs to be updated when neigh_list rebuild
  if (ago == 0) {
    species_t = torch::from_blob(species.data(), {1, ntotal}, torch::dtype(torch::kLong)).to(device);
    species_ghost_as_padding_t = species_t.detach().clone();
    // equivalent to: species_ghost_as_padding[:, nlocal:] = -1
    species_ghost_as_padding_t.index_put_({torch::indexing::Slice(), torch::indexing::Slice(nlocal, torch::indexing::None)}, -1);
    atom_index12_t = torch::from_blob(atom_index12, {2, npairs_half}, torch::dtype(torch::kLong)).to(device);
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

  // run ani model
  auto energy_force = model.forward(inputs).toTuple();

  // extract energy and force from model outputs,
  // and convert the unit to kcal/mol
  auto energy = energy_force->elements()[0].toTensor() * hartree2kcalmol;
  auto force = energy_force->elements()[1].toTensor() * hartree2kcalmol;

  std::cout << "coordinates_t: " << coordinates_t << std::endl;
  std::cout << "species_t: " << species_t << std::endl;
  // std::cout << "atom_index12_t: " << atom_index12_t << std::endl;
  // std::cout << "diff_vector_t: " << diff_vector_t << std::endl;
  // std::cout << "distances_t: " << distances_t << std::endl;
  std::cout << "energy: " << std::setprecision(15) << energy.item<double>() << std::endl;
  std::cout << "force: " << force << std::endl;
  auto in_cutoff = (distances_t <= 5.1).nonzero().flatten();
  // std::cout << "in_cutoff: " << in_cutoff << std::endl;
  distances_t = distances_t.index({in_cutoff});
  // std::cout << "distances_t: " << std::get<0>(distances_t.sort()) << std::endl;

  // write energy and force out
  out_energy = energy.item<double>();
  out_force_t.copy_(force);
}
