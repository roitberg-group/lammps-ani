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

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "update.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairANI::PairANI(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 0;
  npairs = 0;
  npairs_max = 0;
  atom_index12 = nullptr;
  single_enable = 0;
  // require real units, ani model will return energy in kcal/mol
  if (strcmp(update->unit_style, "real") != 0) {
    error->all(FLERR, "Pair ani requires real units");
  }
}

/* ---------------------------------------------------------------------- */

PairANI::~PairANI()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
  memory->destroy(atom_index12);
}

/* ---------------------------------------------------------------------- */

void PairANI::compute(int eflag, int vflag)
{
  if (eflag || vflag) ev_setup(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int ntotal = nlocal + nghost;
  // int newton_pair = force->newton_pair; TODO?

  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  // ani model outputs
  double out_energy;
  std::vector<double> out_force(ntotal * 3);

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
      memory->grow(atom_index12, 2 * npairs_max, "pair:atom_index12");
    }

    // loop over neighbors of local atoms
    int ipair = 0;
    for (int ii = 0; ii < inum; ii++) {
      int i = ilist[ii];
      int *jlist = firstneigh[i];
      int jnum = numneigh[i];

      for (int jj = 0; jj < jnum; jj++) {
        int j = jlist[jj];
        atom_index12[npairs * 0 + ipair] = i;
        atom_index12[npairs * 1 + ipair] = j;
        ipair++;
      }
    }
  }

  // std::cout << "ago: " << ago << ", nlocal :" << nlocal << ", nghost :" <<
  // nghost << ", npairs : " << npairs << std::endl;

  // run ani model
  ani.compute(out_energy, out_force, species, coordinates, npairs, atom_index12, nlocal, ago);

  // write out force
  for (int ii = 0; ii < ntotal; ii++) {
    f[ii][0] += out_force[ii * 3 + 0];
    f[ii][1] += out_force[ii * 3 + 1];
    f[ii][2] += out_force[ii * 3 + 2];
  }

  if (eflag) eng_vdwl += out_energy;
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairANI::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++) setflag[i][j] = 0;

  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

int PairANI::get_local_rank(std::string device_str)
{
  // not the proper way, when try to cast to interger, srun mpi failed
  // const char* nl_rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  // int node_local_rank = atoi(nl_rank);

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


void PairANI::settings(int narg, char **arg)
{
  if (narg < 1) error->all(FLERR, "Illegal pair_style command");

  // read cutoff
  cutoff = utils::numeric(FLERR, arg[0], false, lmp);

  // parsing pairstyle argument
  model_file = arg[1];
  device_str = arg[2];
  int local_rank = get_local_rank(device_str);

  // load model
  ani = ANI(model_file, local_rank);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairANI::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
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
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairANI::init_one(int i, int j)
{
  return cutoff;
}

void *PairANI::extract(const char *str, int &dim)
{
  return nullptr;
}

void PairANI::read_restart(FILE *fp)
{
  // cutoff
  utils::sfread(FLERR, &cutoff, sizeof(double), 1, fp, nullptr, error);

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
  ani = ANI(model_file, local_rank);
}

void PairANI::write_restart(FILE *fp)
{
  // cutoff
  fwrite(&cutoff,sizeof(double),1,fp);

  // TODO fwrite string is bad practice

  // model_file_size device_str_size
  int model_file_size = model_file.size();
  fwrite(&model_file_size,sizeof(int),1,fp);
  int device_str_size = device_str.size();
  fwrite(&device_str_size,sizeof(int),1,fp);

  // model_file device_str
  fwrite(model_file.c_str(),sizeof(char),model_file.size(),fp);
  fwrite(device_str.c_str(),sizeof(char),device_str.size(),fp);
}
