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

  std::vector<float> diff_vector = {-0.1000,  0.7490,  0.5870,
                                    0.5450, -0.6170,  0.4880,
                                    0.0950,  0.8780, -0.3700,
                                    -0.6700,  0.9830, -0.9350,
                                    -0.3680,  0.6630, -0.5840,
                                    0.3790,  0.9720, -1.0970,
                                    0.5950,  0.7230, -0.1990,
                                    0.6450, -1.3660, -0.0990,
                                    -0.3270,  0.6550,  0.6180,
                                    -0.9420, -0.1630,  0.0450,
                                    -1.2690,  0.4920,  0.6630,
                                    0.7050,  0.5570, -0.3300,
                                    -0.7300, -1.2960, -0.2780,
                                    -0.0250, -0.7390, -0.6080,
                                    -0.5100, -0.4890,  0.6460,
                                    0.6700,  1.3380, -0.2340,
                                    0.1600,  0.8490,  0.4120,
                                    0.7470,  0.3090, -0.5130,
                                    0.7650, -0.1050,  0.5650};

  std::vector<float> distances = {0.9569, 0.9570, 0.9575, 1.5131, 0.9571, 1.5139, 0.9573,
                                  1.5139, 0.9581, 0.9571, 1.5139, 0.9572, 1.5132, 0.9573,
                                  0.9574, 1.5146, 0.9572, 0.9574, 0.9568};

  std::vector<int64_t> ghost_index = {};

  // run the model
  double out_energy = 0;
  double out_energy_ref = -534.0368641268269;
  int ntotal = species.size();
  std::vector<float> out_force (ntotal * 3);
  ani.compute(out_energy, out_force, species, coords, atom_index12, diff_vector, distances, ghost_index);
  std::cout << "First call : energy " << out_energy << std::endl;
  std::cout << "First call : force " << out_force[0] << ", " << out_force[1] << ", " << out_force[2] << std::endl;
  TORCH_CHECK(abs(out_energy - out_energy_ref) < 1e-5, "Wrong Energy");
  std::cout << std::endl;

  // set a ghost atom
  ghost_index = {19};
  // reset energy and force
  out_energy = 0;
  out_energy_ref = -533.4612861349258;
  for (auto& f : out_force) {f = 0.f;}
  // run again
  ani.compute(out_energy, out_force, species, coords, atom_index12, diff_vector, distances, ghost_index);
  std::cout << "Second call: energy " << out_energy << std::endl;
  std::cout << "Second call: force " << out_force[0] << ", " << out_force[1] << ", " << out_force[2] << std::endl;
  TORCH_CHECK(abs(out_energy - out_energy_ref) < 1e-5, "Wrong Energy");

  return 0;
}


int main(int argc, const char *argv[]) {
  // test_cuaev(argc, argv);
  test_ani2x_withnbr(argc, argv);
}
