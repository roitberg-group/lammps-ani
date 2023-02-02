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

#ifndef LMP_PAIR_ANI_KOKKOS_H
#define LMP_PAIR_ANI_KOKKOS_H

#include "pair_ani.h"
#include "pair_kokkos.h"

namespace LAMMPS_NS {

template <class DeviceType>
class PairANIKokkos : public PairANI {
 public:
  enum { EnabledNeighFlags = FULL | HALFTHREAD | HALF };
  enum { COUL_FLAG = 0 };
  typedef DeviceType device_type;
  PairANIKokkos(class LAMMPS*);
  ~PairANIKokkos() override;
  void compute(int, int) override;
  // void settings(int, char**) override;
  void init_style() override;
  // double init_one(int, int) override;

 protected:
  typename ArrayTypes<DeviceType>::t_x_array x;
  typename ArrayTypes<DeviceType>::t_f_array f;
  typename ArrayTypes<DeviceType>::t_int_1d type;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename ArrayTypes<DeviceType>::t_efloat_1d d_eatom;
  typename ArrayTypes<DeviceType>::t_virial_array d_vatom;
  typename ArrayTypes<DeviceType>::t_tagint_1d tag;

  typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors; // LayoutRight
  int newton_pair;
  double special_lj[4];

  int neighflag;
  int nlocal, ntotal, eflag, vflag;

  //   void allocate() override;
};
} // namespace LAMMPS_NS

#endif
