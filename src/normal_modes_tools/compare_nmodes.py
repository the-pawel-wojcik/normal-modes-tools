import argparse
import normal_modes_tools as nmt


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""Compare two sets of normal modes."""
    )
    parser.add_argument('old')
    parser.add_argument('new')
    args = parser.parse_args()
    return args

def compare_mode_frequencies(
    ground: list[nmt.NormalMode],
    *targets: list[nmt.NormalMode],
) -> None:
    print("idx    X  A-X  B-X  C-X")
    for idx, (x, *ts) in enumerate(zip(ground, *targets)):
        print(f'{idx:<3d} {x.frequency:4.0f}', end='')
        for t in ts:
            diff = t.frequency - x.frequency
            print(f' {diff:4.0f}', end='')
        print('')

def compare_mode_alignment(
    basis: list[nmt.NormalMode],
    *targets_list: list[nmt.NormalMode],
) -> None:
    for target in targets_list:
        for idx, target_mode in enumerate(target):
            active = list()
            for basis_idx, basis_mode in enumerate(basis):
                inner = basis_mode @ target_mode
                if abs(inner) > 0.032: # sqrt(0.001) aka. 0.1 %
                    active.append({
                        'inner': inner,
                        'basis idx': basis_idx,
                    })
            print(
                f'Mode {idx:<2d} ({target_mode.frequency:4.0f} cm-1):',
                end=''
            )
            for term in sorted(
                active,
                key=lambda x: abs(x['inner']),
                reverse=True
            ):
                inner = term['inner']
                basis_idx = term['basis idx']
                print(f' {inner:+4.3f}|{basis_idx:>2d}>', end='')
            print('')
        print('')


def main():
    # args = get_args()
    # old_nmds_path = args.old
    # new_nmds_path = args.new
    ground_nmds_path = "~/chemistry/cci/phenoxide/calculations/phenoxide/strontium/vib/dz/findiff/normal_modes.xyz"
    # a_nmds_path = "~/chemistry/cci/phenoxide/calculations/phenoxide/strontium/vib/g0a/findiff/nmodes.xyz"
    # b_nmds_path = "~/chemistry/cci/phenoxide/calculations/phenoxide/strontium/vib/g0b/findiff/nmodes_cfour.xyz"
    # c_nmds_path = "~/chemistry/cci/phenoxide/calculations/phenoxide/strontium/vib/g0c/findiff/nmodes_cfour.xyz"
    deuterated_path = "./SrOPh-5d_normal_modes.xyz"

    ground_nmds = nmt.collect_normal_modes(ground_nmds_path)
    # a_nmds = nmt.collect_normal_modes(a_nmds_path)
    # b_nmds = nmt.collect_normal_modes(b_nmds_path)
    # c_nmds = nmt.collect_normal_modes(c_nmds_path)
    deuterated_nmds = nmt.collect_normal_modes(deuterated_path)

    modes = [
        {
            'name': 'SrOPh X',
            'modes': ground_nmds,
        },
        # {
        #     'name': 'A',
        #     'modes': a_nmds,
        # },
        # {
        #     'name': 'B',
        #     'modes': b_nmds,
        # },
        # {
        #     'name': 'C',
        #     'modes': c_nmds,
        # },
        {
            'name': 'SrOPh-5d X',
            'modes': deuterated_nmds,
        },
    ]

    for mode_item in modes:
        print(f'Normal modes of {mode_item['name']} in the basis of normal '
              'modes of X')
        # compare_mode_frequencies(ground_nmds, a_nmds, b_nmds, c_nmds)
        compare_mode_alignment(ground_nmds, mode_item['modes'])


if __name__ == "__main__":
    main()
