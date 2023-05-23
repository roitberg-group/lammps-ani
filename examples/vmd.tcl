# ortho
display projection Orthographic

# del lines
mol delrep 0 0

# dynamic bonds
mol representation DynamicBonds 1.200000 0.200000 12.000000
mol selection all
mol material Opaque
mol addrep 0

# vmd
mol representation VDW 0.200000 12.000000
mol selection all
mol material Opaque
mol addrep 0

# pbc
pbc box
