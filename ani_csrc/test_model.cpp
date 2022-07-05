#include <torch/script.h>
#include <torch/torch.h>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include "ani.h"

int test_cuaev(int argc, const char* argv[]) {
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
  ANI ani{model_file, local_rank};

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

  torch::Tensor species =
      torch::tensor({{1, 0, 0, 0, 0}, {2, 0, 0, 0, -1}}, torch::TensorOptions().requires_grad(false).dtype(torch::kLong));

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

int test_ani2x_withnbr(int argc, const char* argv[]) {
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
  ANI ani{model_file, local_rank};
  std::vector<double> coords = {
      2.0110,  -3.1160, 0.4630,  2.8600,  -3.5250, 0.2940,  2.1650,  -2.1810, 0.3310,  2.3860,  -0.1180, 2.2780,  2.8280,
      0.1650,  3.0780,  2.7810,  0.4120,  1.5850,  1.3800,  1.8550,  0.5400,  1.9420,  2.5970,  0.3170,  1.1310,  2.0080,
      1.4520,  -0.8220, -3.4130, 0.5740,  0.1330,  -3.3460, 0.5680,  -1.1180, -2.5880, 0.9580,  -0.5550, 2.1850,  -2.0950,
      0.0070,  2.8520,  -2.4900, -0.0200, 1.8030,  -1.3990, 2.0700,  -0.4910, -0.6650, 1.7170,  0.3730,  -0.4510, 1.3800,
      -0.9100, -1.1800, -2.2820, 0.7520,  0.2270,  -2.6030, 0.3350,  -0.5730, -2.9100, 0.4930,  0.9010,  -0.2100, -0.8570,
      1.5410,  0.7440,  -0.7800, 1.5740,  -0.4930, -0.1120, 1.0110,  -0.2000, -1.3560, -2.4640, -0.8370, -0.8980, -3.0130,
      -0.7270, -1.9420, -1.9220, -3.1270, 2.2210,  -3.0950, -2.7980, 2.6750,  -3.8710, -2.3830, 2.2020,  -2.4940};
  std::vector<int64_t> species = {3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0};
  std::vector<int64_t> atom_index12 = {
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,
      1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
      2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
      3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,
      5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
      6,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
      8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,
      9,  9,  9,  9,  9,  9,  9,  9,  10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11,
      11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13,
      13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15,
      15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17,
      17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20,
      20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 25, 25,
      25, 25, 26, 26, 26, 27, 27, 28, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
      24, 25, 26, 2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 3,  4,  5,  6,
      7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
      23, 24, 26, 6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 7,  8,  9,  10, 11, 12,
      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
      22, 23, 24, 25, 26, 27, 28, 29, 9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 10, 11,
      12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
      26, 29, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      25, 26, 27, 28, 29, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      25, 26, 27, 28, 29, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
      29, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 20, 21, 22, 23, 24, 25, 26,
      27, 28, 29, 21, 22, 23, 24, 25, 26, 27, 28, 29, 22, 23, 24, 25, 26, 27, 28, 29, 23, 24, 25, 26, 27, 29, 24, 25, 26, 27, 28,
      29, 25, 26, 27, 28, 29, 26, 27, 28, 29, 27, 28, 29, 28, 29, 29};
  std::vector<double> expected_force = {
      -5.4647686256806, -12.5002990030509, 4.7370175310726,  3.7841714273203,   -5.2102273393978,  -1.2301993717466,
      1.1022154207465,  23.5661737917155,  -4.2110574744392, 0.6790959189775,   1.1477697440766,   2.5021304683756,
      2.6530071767145,  -2.4666044603933,  10.8218060184737, 1.3040606344562,   0.1536786267190,   -12.7823759127309,
      -0.6207894274532, 3.6251003503126,   -9.9131096729250, 5.7358496713801,   8.9028091809051,   -4.1886843528134,
      -6.9889654236424, -4.4590944337254,  10.8016518666192, -22.3734053277123, -18.2605912885781, -17.9169685342831,
      16.1228774006969, 1.9230880526167,   2.0465768922730,  8.8881926443038,   15.6500208803591,  8.4692026027651,
      -3.6747574446665, -3.1421083291598,  -4.5969583504707, 3.6822729164539,   6.1974848013216,   -4.2255219317669,
      0.9570170468660,  -4.3808186121820,  12.8469266637504, 16.1929770315060,  -10.4471088710749, 5.3695710147814,
      -4.1349560780569, 7.5275906821907,   7.2651156332145,  -12.8731176857296, -7.2531831162886,  -15.2808435308301,
      13.7511876790221, 7.7336008267141,   -0.1938297516126, -3.2414821809097,  -2.5685872318752,  -8.3234423912134,
      -5.7507978432040, -3.1039358384716,  6.8650908381304,  -12.7381348377138, -4.6380482532534,  2.7585973447672,
      13.9180646982810, -2.8055921170553,  7.1107406496840,  -11.4113737814231, 8.2228405799763,   -9.0294506028059,
      11.1398224857296, 3.1116307417221,   1.1254491661202,  -3.4486710267508,  7.6197902446445,   -3.5608358904941,
      -4.5846767990083, -12.3698440286328, 12.7301342853335, -20.5454285652688, -6.0847390627988,  -2.6413316873809,
      3.2062725294211,  2.9992152156371,   -4.1524747345601, 14.7342403653448,  1.3099882670269,   6.7970732147122};
  double expected_energy = -763.9931790697472 * hartree2kcalmol;

  int nlocal = species.size();

  // run the model
  double out_energy = 0;
  double energy_err;
  double force_err;
  int ntotal = species.size();
  std::vector<double> out_force(ntotal * 3);
  int npairs = atom_index12.size() / 2;

  // run the model
  ani.compute(out_energy, out_force, species, coords, npairs, atom_index12.data(), nlocal);

  // check error
  energy_err = abs(out_energy - expected_energy);
  force_err = 0.0;
  // check force error
  for (int i = 0; i < out_force.size(); i++) {
    auto err = std::abs(expected_force[i] - out_force[i]);
    // std::cout << std::scientific << i << ": " << expected_force[i] << " " << out_force[i] << " " << err << ", " <<
    // std::endl;
    force_err = std::max(force_err, err);
  }

  std::cout << "First call : energy " << std::fixed << out_energy << ", error: " << std::scientific << energy_err << std::endl;
  std::cout << "First call : force " << out_force[0] << ", " << out_force[1] << ", " << out_force[2]
            << ", error: " << std::scientific << force_err << std::endl;
  TORCH_CHECK(energy_err < 1e-5, "Wrong Energy");
  TORCH_CHECK(force_err < 1e-5, "Wrong Forces");
  std::cout << std::endl;

  // // set a ghost atom
  // nlocal = 19;
  // // reset energy and force
  // out_energy = 0;
  // expected_energy = -533.4612861349258 * hartree2kcalmol;
  // for (auto& f : out_force) {f = 0.f;}
  // // run again
  // ani.compute(out_energy, out_force, species, coords, npairs, atom_index12.data(), nlocal);
  // std::cout << "Second call: energy " << std::fixed << out_energy << ", error: " << energy_err << std::endl;
  // std::cout << "Second call: force " << out_force[0] << ", " << out_force[1] << ", " << out_force[2] << std::endl;
  // TORCH_CHECK(energy_err < 1e-5, "Wrong Energy");

  return 0;
}

int main(int argc, const char* argv[]) {
  // test_cuaev(argc, argv);
  test_ani2x_withnbr(argc, argv);
}
