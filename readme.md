# `normal_modes_tool`
Python tools helpful in working with normal modes.

## Capabilities
- Displace the molecular geometry along a selected normal mode, see
  `test/displace/displace.py`.
- Compare two sets of normal modes and their harmonic frequencies, including
  building of the matrix of normal mode's overlaps, i.e., the Duszyński matrix
  `test/sroph/build_duszynski.py`.
- Express the difference between two molecular geometries as displacement along
  normal modes, see `test/pyrazine/find_dQ.py`.
- Calculate the Huang-Rhys factors, see `test/pyrazine/print_HRf.py`.
- Find normal modes for molecule with isotope substitutions, see
  `test/xsim/deuterate_nmodes.py`.
- Transform gradients expressed in one set of normal modes to another, see
  `test/xsim/deuterate_xsim_gradient.py`.
