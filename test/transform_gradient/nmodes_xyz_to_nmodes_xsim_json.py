import normal_modes_tools as nmt
import json

deuterated_xyz_fname = 'outputs/sroph-5d_f0.xyz'


def main():
    deuterated_nmodes = nmt.xyz_file_to_NormalModesList(deuterated_xyz_fname)
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
