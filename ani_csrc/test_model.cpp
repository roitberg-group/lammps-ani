#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <iostream>
#include <vector>
#include <cstdint>
#include "ani.h"


int test_cuaev(int argc, const char *argv[]) {
  // Get the path of the model and the device type
  if (argc != 3) {
    std::cerr << "usage: test_model <model.pt> <cpu/cuda>\n";
    return -1;
  }
  std::string model_file = argv[1];
  std::string device_str = argv[2];
  int local_rank;
  if (device_str != "cpu" && device_str != "cuda") {
    std::cerr << "2nd argument must be <cpu/cuda>\n";
    return -1;
  }
  local_rank = device_str == "cpu" ? -1 : 0;
  ANI ani {model_file, local_rank};

  torch::Tensor coords = torch::tensor(
      {{{0.03192167, 0.00638559, 0.01301679},
        {-0.83140486, 0.39370209, -0.26395324},
        {-0.66518241, -0.84461308, 0.20759389},
        {0.45554739, 0.54289633, 0.81170881},
        {0.66091919, -0.16799635, -0.91037834}},
       {{-4.1862600, 0.0575700, -0.0381200},
        {-3.1689400, 0.0523700, 0.0200000},
        {-4.4978600, 0.8211300, 0.5604100},
        {-4.4978700, -0.8000100, 0.4155600},
        {0.00000000, -0.00000000, -0.00000000}}},
      torch::TensorOptions().requires_grad(true).dtype(torch::kFloat32));

  torch::Tensor species = torch::tensor(
      {{1, 0, 0, 0, 0}, {2, 0, 0, 0, -1}},
      torch::TensorOptions().requires_grad(false).dtype(torch::kLong));

  // Define the input variables
  coords = coords.to(ani.device);
  species = species.to(ani.device);
  std::vector<torch::jit::IValue> inputs;
  std::vector<torch::jit::IValue> tuple;
  tuple.push_back(species);
  tuple.push_back(coords);
  inputs.push_back(torch::ivalue::Tuple::create(tuple));

  // Run the model
  auto aev = ani.model.forward(inputs).toTuple()->elements()[1].toTensor();
  std::cout << "First call :  " << aev.sizes() << std::endl;
  auto aev1 = ani.model.forward(inputs).toTuple()->elements()[1].toTensor();
  std::cout << "Second call: " << aev1.sizes() << std::endl;

  return 0;
}


int test_ani2x_ref(int argc, const char *argv[]) {
  // Get the path of the model and the device type
  if (argc != 3) {
    std::cerr << "usage: test_model <model.pt> <cpu/cuda>\n";
    return -1;
  }
  std::string model_file = argv[1];
  std::string device_str = argv[2];
  int local_rank;
  if (device_str != "cpu" && device_str != "cuda") {
    std::cerr << "2nd argument must be <cpu/cuda>\n";
    return -1;
  }
  local_rank = device_str == "cpu" ? -1 : 0;
  ANI ani {model_file, local_rank};
  torch::Tensor coords = torch::tensor(
      {{{-95.8750, -86.3210, -86.2390},
         {-95.9750, -85.5720, -85.6520},
         {-95.3300, -86.9380, -85.7510},
         {-80.5940, -82.9920, -96.5380},
         {-80.6890, -83.8700, -96.1680},
         {-81.3590, -82.8870, -97.1030},
         {-78.7080, -94.7330, -70.0690},
         {-79.4550, -95.0420, -69.5560},
         {-79.0760, -94.0700, -70.6530},
         {-93.0320, -72.7220, -95.8670},
         {-93.7370, -73.2790, -95.5370},
         {-93.0070, -71.9830, -95.2590},
         {-78.8710, -98.8470, -78.2650},
         {-79.0310, -99.6960, -78.6770},
         {-78.3610, -98.3580, -78.9110},
         {-93.2850, -81.2860, -78.5300},
         {-93.6120, -80.6310, -77.9120},
         {-92.3430, -81.1230, -78.5750},
         {-88.1110, -88.0280, -87.9190},
         {-88.7060, -88.7510, -87.7200}}},
      torch::TensorOptions().requires_grad(true).dtype(torch::kFloat32));

  torch::Tensor species = torch::tensor(
      {{3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0}},
      torch::TensorOptions().requires_grad(false).dtype(torch::kLong));

  // Define the input variables
  coords = coords.to(ani.device);
  species = species.to(ani.device);
  std::vector<torch::jit::IValue> inputs;
  std::vector<torch::jit::IValue> tuple;
  tuple.push_back(species);
  tuple.push_back(coords);
  inputs.push_back(torch::ivalue::Tuple::create(tuple));

  // Run the model
  auto energy = ani.model.forward(inputs).toTuple()->elements()[1].toTensor();
  std::cout << "First call :  " << energy.sizes() << " " << energy << std::endl;
  auto energy1 = ani.model.forward(inputs).toTuple()->elements()[1].toTensor();
  std::cout << "Second call: " << energy1.sizes() << " " << energy1 << std::endl;

  return 0;
}

int test_ani2x_withnbr(int argc, const char *argv[]) {
  // Get the path of the model and the device type
  if (argc != 3) {
    std::cerr << "usage: test_model <model.pt> <cpu/cuda>\n";
    return -1;
  }
  std::string model_file = argv[1];
  std::string device_str = argv[2];
  int local_rank;
  if (device_str != "cpu" && device_str != "cuda") {
    std::cerr << "2nd argument must be <cpu/cuda>\n";
    return -1;
  }
  local_rank = device_str == "cpu" ? -1 : 0;
  ANI ani {model_file, local_rank};
  std::vector<float> coords = {-95.8750, -86.3210, -86.2390,
         -95.9750, -85.5720, -85.6520,
         -95.3300, -86.9380, -85.7510,
         -80.5940, -82.9920, -96.5380,
         -80.6890, -83.8700, -96.1680,
         -81.3590, -82.8870, -97.1030,
         -78.7080, -94.7330, -70.0690,
         -79.4550, -95.0420, -69.5560,
         -79.0760, -94.0700, -70.6530,
         -93.0320, -72.7220, -95.8670,
         -93.7370, -73.2790, -95.5370,
         -93.0070, -71.9830, -95.2590,
         -78.8710, -98.8470, -78.2650,
         -79.0310, -99.6960, -78.6770,
         -78.3610, -98.3580, -78.9110,
         -93.2850, -81.2860, -78.5300,
         -93.6120, -80.6310, -77.9120,
         -92.3430, -81.1230, -78.5750,
         -88.1110, -88.0280, -87.9190,
         -88.7060, -88.7510, -87.7200};

  std::vector<int64_t> species = {3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0};

  std::vector<int64_t> atom_index12 = {1,  2,  3,  5,  8,  8, 18,  2, 16, 15, 16,  9, 10,  9, 12, 14, 12,  6, 3,
                                       0,  0,  4,  4,  6,  7, 19,  1, 15, 17, 17, 10, 11, 11, 14, 13, 13,  7, 5};

  int nlocal = species.size();

  // run the model
  double out_energy = 0;
  double out_energy_ref = -534.0368622716323 * hartree2kcalmol;
  double energy_err;
  int ntotal = species.size();
  std::vector<float> out_force (ntotal * 3);
  int npairs = atom_index12.size() / 2;
  ani.compute(out_energy, out_force, species, coords, npairs, atom_index12.data(), nlocal);
  energy_err = abs(out_energy - out_energy_ref);
  std::cout << "First call : energy " <<  std::fixed << out_energy << ", error: " << energy_err << std::endl;
  std::cout << "First call : force " << out_force[0] << ", " << out_force[1] << ", " << out_force[2] << std::endl;
  TORCH_CHECK(energy_err < 1e-5, "Wrong Energy");
  std::cout << std::endl;

  // set a ghost atom
  nlocal = 19;
  // reset energy and force
  out_energy = 0;
  out_energy_ref = -533.4612861349258 * hartree2kcalmol;
  for (auto& f : out_force) {f = 0.f;}
  // run again
  ani.compute(out_energy, out_force, species, coords, npairs, atom_index12.data(), nlocal);
  std::cout << "Second call: energy " << std::fixed << out_energy << ", error: " << energy_err << std::endl;
  std::cout << "Second call: force " << out_force[0] << ", " << out_force[1] << ", " << out_force[2] << std::endl;
  TORCH_CHECK(energy_err < 1e-5, "Wrong Energy");

  return 0;
}


int main(int argc, const char *argv[]) {
  // test_cuaev(argc, argv);
  test_ani2x_withnbr(argc, argv);
}
