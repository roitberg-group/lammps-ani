
#include "lammpsplugin.h"

#include "version.h"

#include <cstring>

#include "pair_ani.h"

using namespace LAMMPS_NS;

static Pair *ani_creator(LAMMPS *lmp)
{
  return new PairANI(lmp);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "ani";
  plugin.info = "ANI pair style";
  plugin.author = "Jinze Xue (jinzexue@ufl.edu)";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &ani_creator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
