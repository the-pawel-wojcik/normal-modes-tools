import normal_modes_tools as nmt

def main():
    ground_path = './inputs/ccsd-ano1-X.xyz'
    b3u_path = './inputs/eomccsd-ano1-1B3u.xyz'

    ground_nmds = nmt.xyz_file_to_NormalModesList(ground_path)
    b3u_nmds = nmt.xyz_file_to_NormalModesList(b3u_path)

    ground_geo = ground_nmds[0].at
    b3u_geo = b3u_nmds[0].at

    dqs = nmt.find_nmodes_displacement(ground_geo, b3u_geo, ground_nmds)
    for idx, (dq, mode) in enumerate(zip(dqs, ground_nmds)):
        print(f'{idx:2d} ({mode.frequency:4.0f}) {dq:6.2f}')

    
if __name__ == "__main__":
    main()
