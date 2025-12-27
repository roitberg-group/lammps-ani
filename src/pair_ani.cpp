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

#include "pair_ani.h"

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <cmath>
#include <cstring>
#include <vector>
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"

#include <c10/util/env.h>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairANI::PairANI(LAMMPS* lmp) : Pair(lmp) {
  writedata = 0;
  npairs = 0;
  npairs_max = 0;
  atom_index12 = nullptr;
  jlist = nullptr;
  single_enable = 0;
  // require real units, ani model will return energy in kcal/mol
  if (strcmp(update->unit_style, "real") != 0) {
    error->all(FLERR, "Pair ani requires real units");
  }
  comm_reverse = 3;
  comm_reverse_off = 3;
  lammps_ani_profiling = c10::utils::check_env("LAMMPS_ANI_PROFILING") == true;
  std::cout << "LAMMPS_ANI_PROFILING mode: " << lammps_ani_profiling << std::endl;
}

/* ---------------------------------------------------------------------- */

PairANI::~PairANI() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
  memory->destroy(atom_index12);
  memory->destroy(jlist);
}

/* ---------------------------------------------------------------------- */

void PairANI::compute(int eflag, int vflag) {
  if (eflag || vflag)
    ev_setup(eflag, vflag);

  double** x = atom->x;
  double** f = atom->f;
  int* type = atom->type;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int ntotal = nlocal + nghost;
  // newton is on if either newton_pair or newton_bond is on.
  // https://github.com/lammps/lammps/blob/66bbfa67dcbca7dbb81a7be45184233e51022030/src/input.cpp#L1654-L1655
  // if either newton is on, we don't need to mannuly call reverse_comm().
  // https://github.com/lammps/lammps/blob/66bbfa67dcbca7dbb81a7be45184233e51022030/src/verlet.cpp#L340-L343
  int newton = force->newton;

  int inum = list->inum;
  int* ilist = list->ilist;
  int* numneigh = list->numneigh;
  int** firstneigh = list->firstneigh;

  // ani model outputs
  double out_energy;
  out_force.resize(ntotal * 3);
  std::vector<double> out_atomic_energies;
  std::vector<double> out_virial(9);

  // ani model inputs
  std::vector<int64_t> species(ntotal);
  std::vector<double> coordinates(ntotal * 3);

  // coordinates
  for (int ii = 0; ii < ntotal; ii++) {
    coordinates[ii * 3 + 0] = x[ii][0];
    coordinates[ii * 3 + 1] = x[ii][1];
    coordinates[ii * 3 + 2] = x[ii][2];
  }

  int ago = neighbor->ago;

  // convert neighbor list data
  if (ago == 0) {
    // species
    for (int ii = 0; ii < ntotal; ii++) {
      species[ii] = type[ii] - 1; // lammps type from 1 to n
    }

    // calculate the total number of pairs in current domain
    npairs = 0;
    for (int ii = 0; ii < inum; ii++) {
      npairs += numneigh[ii];
    }

    if (npairs > npairs_max) {
      // every time grow 1.5 times larger to avoid reallocate too frequently
      npairs_max = npairs * 1.5;
      if (ani.use_fullnbr) {
        memory->grow(jlist, npairs_max, "pair:jlist");
      } else {
        memory->grow(atom_index12, 2 * npairs_max, "pair:atom_index12");
      }
    }

    // loop over neighbors of local atoms
    int ipair = 0;
    for (int ii = 0; ii < inum; ii++) {
      int i = ilist[ii];
      int* jlist_i = firstneigh[i];
      int jnum = numneigh[i];

      for (int jj = 0; jj < jnum; jj++) {
        int j = jlist_i[jj];
        j &= NEIGHMASK;
        if (ani.use_fullnbr) {
          // full nbrlist
          jlist[ipair] = j;
        } else {
          // half nbrlist
          atom_index12[npairs * 0 + ipair] = i;
          atom_index12[npairs * 1 + ipair] = j;
        }
        // update index
        ipair++;
      }
    }
  }

  std::vector<double>* out_atomic_energies_ptr;
  if (!eflag_atom) {
    out_atomic_energies_ptr = nullptr;
  } else {
    out_atomic_energies.resize(nlocal);
    out_atomic_energies_ptr = &out_atomic_energies;
  }

  // run ani model
  if (ani.use_fullnbr) {
    ani.compute(
        out_energy,
        out_force,
        out_virial,
        species,
        coordinates,
        npairs,
        ilist, // full nbrlist
        jlist,
        numneigh,
        nlocal,
        ago,
        out_atomic_energies_ptr,
        vflag);
  } else {
    ani.compute(
        out_energy,
        out_force,
        out_virial,
        species,
        coordinates,
        npairs,
        atom_index12, // half nbrlist
        nlocal,
        ago,
        out_atomic_energies_ptr,
        vflag);
  }

  // When newton is off, there will be no reverse communication of the ghost atom's force to the neighboring
  // domain's local atom, so we need to manually call reverse communication.
  // We accumulate forces on out_force instead of f, because when newton flag is off, the previous step's
  // ghost atoms' forces are not cleared.
  // https://github.com/lammps/lammps/blob/66bbfa67dcbca7dbb81a7be45184233e51022030/src/verlet.cpp#L382-L384
  if (!newton) {
    ::nvtxRangePushA("reverse_comm");
    comm->reverse_comm(this);
    ::nvtxRangePop();
  }

  // write out force
  for (int ii = 0; ii < ntotal; ii++) {
    // notes: at this point, ghost atoms' forces have wrong results because they were not cleard between steps,
    // but we don't care because they are not used anyway as long as newton flag is off.
    f[ii][0] += out_force[ii * 3 + 0];
    f[ii][1] += out_force[ii * 3 + 1];
    f[ii][2] += out_force[ii * 3 + 2];
  }

  if (eflag) {
    eng_vdwl += out_energy;
  }

  // write out atomic energies
  if (eflag_atom) {
    for (int ii = 0; ii < nlocal; ++ii) {
      eatom[ii] += out_atomic_energies[ii];
    }
  }

  // write out virial
  if (vflag) {
    virial[0] += out_virial[0 * 3 + 0];
    virial[1] += out_virial[1 * 3 + 1];
    virial[2] += out_virial[2 * 3 + 2];
    virial[3] += out_virial[0 * 3 + 1];
    virial[4] += out_virial[0 * 3 + 2];
    virial[5] += out_virial[1 * 3 + 2];
  }

}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairANI::allocate() {
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

int PairANI::get_local_rank(std::string device_str) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int num_devices = 0;
  cudaGetDeviceCount(&num_devices);

  // get local rank to set cuda device
  int local_rank = -1;
  {
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
  }
  if (num_devices > 0) {
    // map multiple process to same cuda device if there are more ranks
    local_rank = local_rank % num_devices;
  } else {
    local_rank = -1;
  }

  if (device_str != "cpu" && device_str != "cuda") {
    std::cerr << "2nd argument must be <cpu/cuda>\n";
  }
  if (device_str == "cpu") {
    local_rank = -1;
  }
  return local_rank;
}

void PairANI::settings(int narg, char** arg) {
  if (narg < 1)
    error->all(FLERR, "Illegal pair_style command");

  // read cutoff
  // TODO we could read cutoff from the model
  cutoff = utils::numeric(FLERR, arg[0], false, lmp);

  // parsing pairstyle argument
  model_file = arg[1];
  device_str = arg[2];
  int local_rank = get_local_rank(device_str); // -1 for cpu
  use_num_models = narg > 3 ? utils::inumeric(FLERR, arg[3], false, lmp) : -1; // -1 to use all models
  bool use_cuaev, use_fullnbr, use_single;
  // cuaev (default) or pyaev
  if (narg > 4) {
    std::string aev_str = arg[4];
    if (aev_str == "cuaev") {
      use_cuaev = true;
    } else if (aev_str == "pyaev") {
      use_cuaev = false;
    } else {
      error->all(FLERR, "ani_aev should be cuaev or pyaev");
    }
  } else {
    use_cuaev = true;
  }
  // full_nbr (default) or half_nbr
  if (narg > 5) {
    std::string nbr_str = arg[5];
    if (nbr_str == "full") {
      use_fullnbr = true;
    } else if (nbr_str == "half") {
      use_fullnbr = false;
    } else {
      error->all(FLERR, "ani_neighbor should be full or half");
    }
  } else {
    use_fullnbr = true;
  }
  // single (default) or double precision
  if (narg > 6) {
    std::string nbr_str = arg[6];
    if (nbr_str == "single") {
      use_single = true;
    } else if (nbr_str == "double") {
      use_single = false;
    } else {
      error->all(FLERR, "precision should be single or double");
    }
  } else {
    use_single = true;
  }

  // load model
  ani = ANI(model_file, local_rank, use_num_models, use_cuaev, use_fullnbr, use_single);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairANI::coeff(int narg, char** arg) {
  if (!allocated)
    allocate();
  if (narg != 2) {
    error->all(FLERR, "Incorrect args for pair coefficients, it should be set as: pair_coeff * *");
  }

  int ilo, ihi, jlo, jhi;
  int n = atom->ntypes;
  utils::bounds(FLERR, arg[0], 1, n, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, n, jlo, jhi, error);
  if (ilo != 1 || jlo != 1 || ihi != n || jhi != n) {
    error->all(FLERR, "Incorrect args for pair coefficients, it should be set as: pair_coeff * *");
  }

  // setflag
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      setflag[i][j] = 1;
    }
  }
}

/* ----------------------------------------------------------------------
   init specific to a pair style
------------------------------------------------------------------------- */

void PairANI::init_style() {
  // when using half neighbor list, newton_pair must be set as off so that the local atoms have the
  // complete set of neighboring ghost atoms.
  if (!ani.use_fullnbr && force->newton_pair == 1)
    error->all(FLERR, "Pair style ANI requires newton pair off when using half neighbor list");

  // For consistency, we also require newton_pair is off for full nbrlist.
  if (ani.use_fullnbr && force->newton_pair == 1)
    error->all(FLERR, "Pair style ANI requires newton pair off when using full neighbor list");

  if (ani.use_fullnbr) {
    neighbor->add_request(this, NeighConst::REQ_FULL);
  } else {
    // request half neighbor list
    neighbor->add_request(this);
  }
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairANI::init_one(int i, int j) {
  return cutoff;
}

void* PairANI::extract(const char* str, int& dim) {
  return nullptr;
}

/* ----------------------------------------------------------------------
   read and write restart file
------------------------------------------------------------------------- */

void PairANI::read_restart(FILE* fp) {
  // cutoff
  utils::sfread(FLERR, &cutoff, sizeof(double), 1, fp, nullptr, error);
  // use_num_models
  utils::sfread(FLERR, &use_num_models, sizeof(int), 1, fp, nullptr, error);
  // use_cuaev, use_fullnbr and use_single
  bool use_cuaev, use_fullnbr, use_single;
  utils::sfread(FLERR, &use_cuaev, sizeof(bool), 1, fp, nullptr, error);
  utils::sfread(FLERR, &use_fullnbr, sizeof(bool), 1, fp, nullptr, error);
  utils::sfread(FLERR, &use_single, sizeof(bool), 1, fp, nullptr, error);

  // model_file_size device_str_size
  int model_file_size, device_str_size;
  utils::sfread(FLERR, &model_file_size, sizeof(int), 1, fp, nullptr, error);
  utils::sfread(FLERR, &device_str_size, sizeof(int), 1, fp, nullptr, error);
  model_file.resize(model_file_size);
  device_str.resize(device_str_size);

  // model_file device_str
  utils::sfread(FLERR, &model_file[0], sizeof(char), model_file_size, fp, nullptr, error);
  utils::sfread(FLERR, &device_str[0], sizeof(char), device_str_size, fp, nullptr, error);

  // init model
  int local_rank = get_local_rank(device_str);
  ani = ANI(model_file, local_rank, use_num_models, use_cuaev, use_fullnbr, use_single);
}

void PairANI::write_restart(FILE* fp) {
  // cutoff
  fwrite(&cutoff, sizeof(double), 1, fp);
  // use_num_models
  fwrite(&use_num_models, sizeof(int), 1, fp);
  // use_cuaev and use_fullnbr
  fwrite(&ani.use_cuaev, sizeof(bool), 1, fp);
  fwrite(&ani.use_fullnbr, sizeof(bool), 1, fp);
  fwrite(&ani.use_single, sizeof(bool), 1, fp);

  // TODO fwrite string is a bad practice
  // model_file_size device_str_size
  int model_file_size = model_file.size();
  fwrite(&model_file_size, sizeof(int), 1, fp);
  int device_str_size = device_str.size();
  fwrite(&device_str_size, sizeof(int), 1, fp);

  // model_file device_str
  fwrite(model_file.c_str(), sizeof(char), model_file.size(), fp);
  fwrite(device_str.c_str(), sizeof(char), device_str.size(), fp);
}

/* ----------------------------------------------------------------------
   pack and unpack reverse communication
------------------------------------------------------------------------- */

int PairANI::pack_reverse_comm(int n, int first, double* buf) {
  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = out_force[i * 3 + 0];
    buf[m++] = out_force[i * 3 + 1];
    buf[m++] = out_force[i * 3 + 2];
  }
  return m;
}

void PairANI::unpack_reverse_comm(int n, int* list, double* buf) {
  int i, j, m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    out_force[j * 3 + 0] += buf[m++];
    out_force[j * 3 + 1] += buf[m++];
    out_force[j * 3 + 2] += buf[m++];
  }
}

// TODO memory_usage
