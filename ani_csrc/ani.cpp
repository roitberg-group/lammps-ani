#include "ani.h"
#include <nvToolsExt.h>
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

    // std::cout << model.dump_to_str(false, false, false) << std::endl;
    // dummy_buffer
    bool found_dummy_buffer = false;
    for (const torch::jit::NameTensor& p : model.named_buffers(/*recurse=*/false)) {
      if (p.name == "dummy_buffer") {
        dtype = p.value.scalar_type();
        found_dummy_buffer = true;
      }
    }
    TORCH_CHECK(
        found_dummy_buffer,
        "dummy_buffer is not found in your model, please register one with: "
        "self.register_buffer('dummy_buffer', torch.empty(0))");

    // use_fullnbr
    TORCH_CHECK(model.hasattr("use_fullnbr"), "use_fullnbr (bool) is not found in your model");
    use_fullnbr = model.attr("use_fullnbr").toBool();
    std::string nbrlist = use_fullnbr ? "full" : "half";

    // TODO we need to disable nvfuser
    // TORCH_CHECK(model.hasattr("nvfuser_enabled"), "nvfuser_enabled (bool) is not found in your model");
    // bool nvfuser_enabled = model.attr("nvfuser_enabled").toBool();
    // std::cout << "nvfuser_enabled: " << nvfuser_enabled << std::endl;
    // torch::jit::fuser::cuda::setEnabled(nvfuser_enabled);
    torch::jit::setGraphExecutorOptimize(false);

    std::cout << "Successfully loaded the model \nfile: '" << model_file << "' \ndevice: " << device << " \ndtype: " << dtype
              << " \nnbrlist: " << nbrlist << std::endl
              << std::endl;
  } catch (const c10::Error& e) {
    std::cerr << "Error loading the model '" << model_file << "' on " << device << ". " << e.what();
    throw std::runtime_error("Error");
  }
}

// compute with half nbrlist
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
    ::nvtxMarkA("neighbor list rebuilt");
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

// compute with full nbrlist
void ANI::compute(
    double& out_energy,
    std::vector<double>& out_force,
    std::vector<int64_t>& species,
    std::vector<double>& coordinates,
    int npairs,
    int* ilist_unique,
    int* jlist,
    int* numneigh,
    int nlocal,
    int ago,
    std::vector<double>* out_atomic_energies) {
  int ntotal = species.size();

  // output tensor
  auto out_force_t = torch::from_blob(out_force.data(), {1, ntotal, 3}, torch::dtype(torch::kFloat64));
  // input tensor
  auto coordinates_t =
      torch::from_blob(coordinates.data(), {1, ntotal, 3}, torch::dtype(torch::kFloat64)).to(device).requires_grad_(true);

  // species_t, ilist_unique_t, jlist_t and numneigh_t are cloned/cached on devices and only needs to be updated when neigh_list
  // rebuild
  if (ago == 0) {
    // nbrlist
    ilist_unique_t = torch::from_blob(ilist_unique, {nlocal}, torch::dtype(torch::kInt32)).to(device);
    jlist_t = torch::from_blob(jlist, {npairs}, torch::dtype(torch::kInt32)).to(device);
    numneigh_t = torch::from_blob(numneigh, {nlocal}, torch::dtype(torch::kInt32)).to(device);

    species_t = torch::from_blob(species.data(), {1, ntotal}, torch::dtype(torch::kLong)).to(device);
    // when runing on the CPU, we have to explicitly clone this tensor
    // because they are created from temporary vector data pointers
    if (device == torch::kCPU) {
      species_t = species_t.clone();
    }
    species_ghost_as_padding_t = species_t.detach().clone();
    // equivalent to: species_ghost_as_padding[:, nlocal:] = -1
    species_ghost_as_padding_t.index_put_({torch::indexing::Slice(), torch::indexing::Slice(nlocal, torch::indexing::None)}, -1);
  }

  // pack forward inputs
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(species_t);
  inputs.push_back(coordinates_t);
  inputs.push_back(ilist_unique_t);
  inputs.push_back(jlist_t);
  inputs.push_back(numneigh_t);
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
