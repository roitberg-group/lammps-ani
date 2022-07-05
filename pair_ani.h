/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/ Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_PAIR_ANI_H
#define LMP_PAIR_ANI_H

#include "ani.h"
#include "pair.h"

namespace LAMMPS_NS {

class PairANI : public Pair {
 public:
  PairANI(class LAMMPS*);
  ~PairANI() override;
  void compute(int, int) override;

  void settings(int, char**) override;
  void coeff(int, char**) override;
  void init_style() override;
  double init_one(int, int) override;
  void* extract(const char*, int&) override;
  void write_restart(FILE*) override;
  void read_restart(FILE*) override;

  int get_local_rank(std::string device_str);

 protected:
  double cutoff;
  ANI ani;
  int64_t* atom_index12;
  int npairs; // number of pairs in the current domain before neigh_list rebuild
  int npairs_max; // if exceed this max number the allocated atom_index12 needs to grow
  std::string model_file;
  std::string device_str;

  virtual void allocate();
};

} // namespace LAMMPS_NS

#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

*/
