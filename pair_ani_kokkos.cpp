/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "pair_ani_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "math_const.h"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"

#include <torch/script.h>
#include <torch/torch.h>
#include <cmath>
#include <cstring>
#include <type_traits>

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

template <class DeviceType>
PairANIKokkos<DeviceType>::PairANIKokkos(LAMMPS* lmp) : PairANI(lmp) {
  respa_enable = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos*)atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
PairANIKokkos<DeviceType>::~PairANIKokkos() {
  if (copymode)
    return;

  if (allocated) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->destroy_kokkos(k_vatom, vatom);
    // eatom = NULL;
    // vatom = NULL;
  }
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void PairANIKokkos<DeviceType>::compute(int eflag_in, int vflag_in) {
  if (lammps_ani_profiling) {
    torch::cuda::synchronize();
  }

  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL)
    no_virial_fdotr_compute = 1;

  ev_init(eflag, vflag, 0);

  // reallocate per-atom arrays if necessary

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->create_kokkos(k_eatom, eatom, maxeatom, "pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom, vatom);
    memoryKK->create_kokkos(k_vatom, vatom, maxvatom, "pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  atomKK->sync(execution_space, datamask_read);
  if (eflag || vflag)
    atomKK->modified(execution_space, datamask_modify);
  else
    atomKK->modified(execution_space, F_MASK);

  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  tag = atomKK->k_tag.view<DeviceType>();
  nlocal = atom->nlocal;
  ntotal = atom->nlocal + atom->nghost;
  newton_pair = force->newton_pair;
  int ago = neighbor->ago;

  // copymode = 1;
  // TODO we could save some computations and only run when ago == 0

  const int inum = list->inum;
  const int ignum = inum + list->gnum;
  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  auto d_numneigh = k_list->d_numneigh;
  auto d_ilist = k_list->d_ilist;
  // d_neighbors = Kokkos::create_mirror(k_list->d_neighbors);
  d_neighbors = k_list->d_neighbors;

  using FloatView2D = Kokkos::View<float**, Kokkos::LayoutRight, DeviceType>;
  using UnmanagedFloatView1D = Kokkos::View<float*, Kokkos::LayoutRight, DeviceType>;
  using UnmanagedFloatView2D = Kokkos::View<float**, Kokkos::LayoutRight, DeviceType>;
  FloatView2D d_xfloat;

  int max_neighs = d_neighbors.extent(1);

  // Check https://github.com/roitberg-group/lammps-ani/pull/49 for Kokkos data views shape
  // information of: x, f, type, d_neighbors, d_ilist, d_numneigh
  torch::Tensor out_energy, out_force, out_atomic_energies;
  torch::Tensor species, coordinates, ilist_unique, jlist, numneigh;

  auto kokkos_device = torch::Device(torch::kCPU);
  if (std::is_same<DeviceType, LMPHostType>::value && !std::is_same<DeviceType, LMPDeviceType>::value) {
    kokkos_device = torch::kCPU;
  } else {
    // TODO local_rank == -1 is cpu?
    signed char local_rank = Kokkos::Impl::CudaInternal::singleton().m_cudaDev;
    kokkos_device = torch::Device({torch::kCUDA, local_rank});
  }
  // std::cout << "kokkos_device: " << kokkos_device << std::endl;

  typedef typename decltype(x)::non_const_value_type value_type;
  auto force_dytpe = torch::kFloat64;
  if (!std::is_same<value_type, double>::value) {
    // std::cout << "build with float" << std::endl;
    force_dytpe = torch::kFloat32;
  }

  auto tensor_kokkos_int32_option = torch::TensorOptions().dtype(torch::kInt32).device(kokkos_device);
  auto tensor_kokkos_force_option = torch::TensorOptions().dtype(force_dytpe).device(kokkos_device);
  auto tensor_kokkos_float64_option = torch::TensorOptions().dtype(torch::kFloat64).device(kokkos_device);
  species = torch::from_blob(type.data(), {1, ntotal}, tensor_kokkos_int32_option)
                .to(torch::TensorOptions().dtype(torch::kInt64).device(ani.device));
  // lammps type from 1 to n
  species = species - 1;
  coordinates = torch::from_blob(x.data(), {1, ntotal, 3}, tensor_kokkos_force_option).to(ani.device);
  ilist_unique = torch::from_blob(d_ilist.data(), {nlocal}, tensor_kokkos_int32_option).to(ani.device);
  numneigh = torch::from_blob(d_numneigh.data(), {nlocal}, tensor_kokkos_int32_option).to(ani.device);

  // transpose jlist if it is LayoutLeft (column-major)
  typedef typename decltype(d_neighbors)::array_layout d_neighbors_layout;
  int kokkos_ntotal = d_neighbors.extent(0);
  if (std::is_same<d_neighbors_layout, Kokkos::LayoutLeft>::value) {
    // std::cout << "d_neighbors layout == LayoutLeft" << std::endl;
    jlist = torch::from_blob(d_neighbors.data(), {max_neighs, kokkos_ntotal}, tensor_kokkos_int32_option).to(ani.device);
    jlist = jlist.transpose(0, 1);
  } else {
    // std::cout << "d_neighbors layout == LayoutRight" << std::endl;
    jlist = torch::from_blob(d_neighbors.data(), {kokkos_ntotal, max_neighs}, tensor_kokkos_int32_option).to(ani.device);
  }
  // TODO, because of the current API design, we have to flatten the jlist and remove the padding
  jlist = jlist.index({torch::indexing::Slice(0, nlocal), torch::indexing::Slice()});
  torch::Tensor mask = torch::arange(max_neighs, ani.device).unsqueeze(0) < numneigh.unsqueeze(1);
  jlist = jlist.masked_select(mask);
  int npairs = jlist.size(0);

  ani.compute(
      out_energy,
      out_force,
      species,
      coordinates,
      npairs,
      ilist_unique,
      jlist,
      numneigh,
      nlocal,
      ago,
      out_atomic_energies,
      eflag_atom);

  torch::Tensor d_force_tensor = torch::from_blob(f.data(), {1, ntotal, 3}, tensor_kokkos_force_option);
  d_force_tensor += out_force.to(kokkos_device);

  if (eflag) {
    eng_vdwl += out_energy.item<double>();
  }
  if (eflag_atom) {
    torch::Tensor d_eatom_tensor = torch::from_blob(d_eatom.data(), {1, nlocal}, tensor_kokkos_float64_option);
    d_eatom_tensor += out_atomic_energies.to(kokkos_device);
  }

  if (lammps_ani_profiling) {
    torch::cuda::synchronize();
  }
  // copymode = 0;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void PairANIKokkos<DeviceType>::init_style() {
  // PairANI::init_style();
  neighbor->add_request(this, NeighConst::REQ_FULL);

  neighflag = lmp->kokkos->neighflag;
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same<DeviceType, LMPHostType>::value && !std::is_same<DeviceType, LMPDeviceType>::value);
  request->set_kokkos_device(std::is_same<DeviceType, LMPDeviceType>::value);
  // TODO requires full neighbor list and newton on
  if (neighflag != FULL || !ani.use_fullnbr)
    error->all(FLERR, "Pair style ANI requires full neighbor list when using kokkos");
  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style ANI requires newton pair on when using kokkos");
}
/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairANIKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairANIKokkos<LMPHostType>;
#endif
} // namespace LAMMPS_NS
