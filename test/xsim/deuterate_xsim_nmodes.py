import normal_modes_tools as nmt
import json

xyz_filename = 'sroph_f0.xyz'


def main():
    nmodes = nmt.xyz_file_to_NormalModesList(xyz_filename)
    deuterated_nmodes = nmt.deuterate_modes(nmodes)

    xsim_json_nmodes = list()
    for mode in deuterated_nmodes:
        coordinate = [displ.xyz for displ in mode.displacement]
        xsim_json_mode = {
            'symmetry': '???',
            'frequency, cm-1': float(mode.frequency),
            'kind': 'VIBRATION',
            'coordinate': coordinate,
        }
        xsim_json_nmodes.append(xsim_json_mode)

    print(json.dumps(xsim_json_nmodes))

if __name__ == "__main__":
    main()
