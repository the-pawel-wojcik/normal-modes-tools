# `normal_modes_tool`
Python tools helpful in working with normal modes.

## Capabilities
- List normal modes and their frequencies
```bash
# Works only if the comment line in the xyz files has the following format
# 0. Vibration mode, 64.77 cm-1, B1
# i.e., frequency is the fourth whitespace-separated item of the comment line
python -m normal_modes_tool --list nmodes.xyz
```
- Displace the molecular geometry along a selected normal mode, see
  `test/displace/displace.py`.
- Compare two sets of normal modes and their harmonic frequencies, including
  building of the matrix of normal mode's overlaps, i.e., the Duszyński matrix
  `test/sroph/build_duszynski.py`.
```bash
python -m normal_modes_tool --compare new_nmodes.xyz old_nmodes.xyz
```
- Express the difference between two molecular geometries as displacement along
  normal modes, see `test/pyrazine/find_dQ.py`.
- Calculate the Huang-Rhys factors, see `test/pyrazine/print_HRf.py` or
  `test/sroph/hrfs/huang_rhys_factors.py`
- Find normal modes for molecule with isotope substitutions, see
  `test/xsim/deuterate_nmodes.py`.
- Transform gradients expressed in one set of normal modes to another, see
  `test/xsim/deuterate_xsim_gradient.py`.


