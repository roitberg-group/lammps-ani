#include "ani.h"
#include <nvToolsExt.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <vector>

// Modified from the following URL, so it **only** casts floating point parameters and buffers, and skips non-floating ones.
// https://github.com/pytorch/pytorch/blob/1237cf6b6ca86ac6afd5c0a8d3075c9a2d85b6e4/torch/csrc/jit/api/module.cpp#L166-L190
void module_state_to(const torch::autograd::Variable& variable, const at::ScalarType& dtype) {
  // Need to access the `at::Tensor` as a `Variable` here.
  auto new_data = variable.to(
      variable.device(),
      dtype,
      /*non_blocking=*/false);
  variable.set_data(new_data);
}

void module_to_dtype(torch::jit::script::Module model, const at::ScalarType& dtype) {
  for (at::Tensor e : model.parameters()) {
    if (e.is_floating_point()) {
      module_state_to(e, dtype);
    }
  }
  for (at::Tensor e : model.buffers()) {
    if (e.is_floating_point()) {
      module_state_to(e, dtype);
    }
  }
}

ANI::ANI(const std::string& model_file, int local_rank, int use_num_models, bool use_cuaev_, bool use_fullnbr_, bool use_single_)
    : device(local_rank == -1 ? torch::kCPU : torch::kCUDA, local_rank),
      use_cuaev(use_cuaev_),
      use_fullnbr(use_fullnbr_),
      use_single(use_single_) {
  at::globalContext().setAllowTF32CuBLAS(false);
  at::globalContext().setAllowTF32CuDNN(false);
  try {
    model = torch::jit::load(model_file, device);

    // std::cout << model.dump_to_str(false, false, false) << std::endl;

    // change precision
    // Torchscript Module API Reference: https://pytorch.org/cppdocs/api/structtorch_1_1jit_1_1_module.html
    dtype = use_single ? torch::kFloat32 : torch::kFloat64;
    module_to_dtype(model, dtype);

    // prepare inputs
    std::vector<torch::jit::IValue> init_inputs;
    init_inputs.push_back(use_cuaev);
    init_inputs.push_back(use_fullnbr);
    model.get_method("init")(init_inputs);

    std::string ani_aev = use_cuaev ? "cuaev" : "pyaev";
    std::string nbrlist = use_fullnbr ? "full" : "half";

    // select_models
    int num_models = model.attr("num_models").toInt();
    // use all models if not specified (-1)
    if (use_num_models == -1) {
      use_num_models = num_models;
    }
    // prepare inputs
    std::vector<torch::jit::IValue> select_models_inputs;
    select_models_inputs.push_back(use_num_models);
    // select models
    model.get_method("select_models")(select_models_inputs);

    // nvfuser and graph optimization are disabled
    torch::jit::fuser::cuda::setEnabled(false);
    torch::jit::setGraphExecutorOptimize(false);

    std::cout << "Successfully loaded the model \nfile: '" << model_file << "' \ndevice: " << device << " \ndtype: " << dtype
              << " \nnbrlist: " << nbrlist << " \nani_aev: " << ani_aev
              << " \nuse_num_models: " << model.attr("use_num_models").toInt() << "/" << model.attr("num_models").toInt()
              << std::endl
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
      torch::from_blob(coordinates.data(), {1, ntotal, 3}, torch::dtype(torch::kFloat64)).to(dtype).to(device).requires_grad_(true);

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
      torch::from_blob(coordinates.data(), {1, ntotal, 3}, torch::dtype(torch::kFloat64)).to(dtype).to(device).requires_grad_(true);

  // species_t, ilist_unique_t, jlist_t and numneigh_t are cloned/cached on devices and only needs to be updated when neigh_list
  // rebuild
  if (ago == 0) {
    // nbrlist
    ::nvtxMarkA("neighbor list rebuilt");
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
  bool eflag_atom = out_atomic_energies != nullptr;
  inputs.push_back(eflag_atom);

  // run ani model
  torch::Tensor energy, force, atomic_energies;
  auto outputs = model.forward(inputs).toTuple();
  // extract energy and force from model outputs, and convert the unit to kcal/mol
  energy = outputs->elements()[0].toTensor() * hartree2kcalmol;
  force = outputs->elements()[1].toTensor() * hartree2kcalmol;

  // write energy and force out
  out_energy = energy.item<double>();
  out_force_t.copy_(force);

  // if eflag_atom is false, atomic_energies will be an empty tensor
  if (eflag_atom) {
    atomic_energies = outputs->elements()[2].toTensor() * hartree2kcalmol;
    auto out_atomic_energies_t = torch::from_blob(out_atomic_energies->data(), {1, nlocal}, torch::dtype(torch::kFloat64));
    out_atomic_energies_t.copy_(atomic_energies);
  }
}

// kokkos compute with full nbrlist
void ANI::compute(
    torch::Tensor& out_energy,
    torch::Tensor& out_force,
    torch::Tensor& species,
    torch::Tensor& coordinates,
    int npairs, // TODO remove?
    torch::Tensor& ilist_unique,
    torch::Tensor& jlist,
    torch::Tensor& numneigh,
    int nlocal,
    int ago,
    torch::Tensor& out_atomic_energies,
    bool eflag_atom) {
  int ntotal = species.size(0);

  torch::Tensor species_ghost_as_padding = species.detach().clone();
  // TODO we don't need this parameter, we could use species and nlocal to create this tensor within python code
  // equivalent to: species_ghost_as_padding[:, nlocal:] = -1
  species_ghost_as_padding.index_put_({torch::indexing::Slice(), torch::indexing::Slice(nlocal, torch::indexing::None)}, -1);

  // pack forward inputs
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(species);
  inputs.push_back(coordinates.to(dtype).requires_grad_(true));
  inputs.push_back(ilist_unique);
  inputs.push_back(jlist);
  inputs.push_back(numneigh);
  inputs.push_back(species_ghost_as_padding);
  inputs.push_back(eflag_atom);

  // run ani model
  torch::Tensor energy, force, atomic_energies;
  auto outputs = model.forward(inputs).toTuple();
  // extract energy and force from model outputs, and convert the unit to kcal/mol
  out_energy = outputs->elements()[0].toTensor() * hartree2kcalmol;
  out_force = outputs->elements()[1].toTensor() * hartree2kcalmol;

  // if eflag_atom is false, atomic_energies will be an empty tensor
  if (eflag_atom) {
    out_atomic_energies = outputs->elements()[2].toTensor() * hartree2kcalmol;
  }
}
