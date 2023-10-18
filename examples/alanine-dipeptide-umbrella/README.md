
we could check the @phi-2 atom indices using the command:
```
MOLINFO STRUCTURE=alanine-dipeptide.vacuum.pdb
phi: TORSION ATOMS=@phi-2
psi: TORSION ATOMS=@psi-2
# phi2: TORSION ATOMS=5,7,9,15
# psi2: TORSION ATOMS=7,9,15,17
# PRINT ARG=phi,psi,phi2,psi2 STRIDE=100 FILE=angles.dat

bb: RESTRAINT ARG=phi KAPPA=100.0 AT=0.0
PRINT ARG=phi,psi,bb.bias FILE=colvar.dat STRIDE=100
```
