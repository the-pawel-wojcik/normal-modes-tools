import normal_modes_tools as nmt

xyz_filename = 'inputs/sroph_f0.xyz'


def main():
    nmodes = nmt.xyz_file_to_NormalModesList(xyz_filename)
    deuterated_nmodes = nmt.deuterate_modes(nmodes)
    print(nmt.normalModesList_to_xyz_file(deuterated_nmodes))

if __name__ == "__main__":
    main()
