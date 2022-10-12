
#include "lammpsplugin.h"

#include "version.h"

#include <cstring>

#include "pair_ani.h"
#include "pair_ani_kokkos.h"

using namespace LAMMPS_NS;

static Pair* ani_creator(LAMMPS* lmp) {
  return new PairANI(lmp);
}

static Pair* ani_kokkos_creator(LAMMPS* lmp) {
  return new PairANIKokkos<LMPDeviceType>(lmp);
}

extern "C" void lammpsplugin_init(void* lmp, void* handle, void* regfunc) {
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc)regfunc;

  // register plain ani pair style
  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "ani";
  plugin.info = "ANI pair style v0.1";
  plugin.author = "Jinze Xue (jinzexue@ufl.edu)";
  plugin.creator.v1 = (lammpsplugin_factory1*)&ani_creator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);

  // also register ani/kk pair style. only need to update changed fields
  plugin.name = "ani/kk";
  plugin.info = "ANI pair style for Kokkos v0.1";
  plugin.creator.v1 = (lammpsplugin_factory1*)&ani_kokkos_creator;
  (*register_plugin)(&plugin, lmp);
}
