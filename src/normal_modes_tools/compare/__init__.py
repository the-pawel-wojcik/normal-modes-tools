import argparse
import normal_modes_tools as nmt
import matplotlib.pyplot as plt
import numpy as np


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""Compare two sets of normal modes."""
    )
    parser.add_argument('old')
    parser.add_argument('new', nargs='+')
    parser.add_argument(
        '--print_freq_comparison',
        default=False,
        action='store_true',
        help='Print frequencies of both sets of modes side by side sorted.',
    )
    parser.add_argument(
        '--print_decomposition',
        default=False,
        action='store_true',
        help='Check if modes are aligned',
    )
    parser.add_argument(
        '--show_duszynski',
        default=False,
        action='store_true',
        help='Display matrix of normal modes overlaps (the Duszyński matrix).',
    )
    args = parser.parse_args()
    return args


def mode_frequencies(
    ground: list[nmt.NormalMode],
    target: list[nmt.NormalMode],
) -> None:
    print("mode freq diff")
    for idx, (x, t) in enumerate(zip(ground, target)):
        print(f'{idx:<3d} {x.frequency:4.0f}', end='')
        diff = t.frequency - x.frequency
        print(f' {diff:4.0f}', end='')
        print('')


def mode_alignment(
    basis_modes: list[nmt.NormalMode],
    target_modes: list[nmt.NormalMode],
    print_threshold: float = 0.032,
) -> None:
    """Decompose `target_modes` in the basis of `basis_modes`.
    `print_threshold` = 0.001 ** 0.5.
    """
    for idx, target_mode in enumerate(target_modes):
        active = list()
        for basis_idx, basis_mode in enumerate(basis_modes):
            inner = basis_mode @ target_mode
            if abs(inner) > print_threshold:
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


def show_duszynski(
    old_nmds: list[nmt.NormalMode],
    new_nmds: list[nmt.NormalMode],
    ax = None,
):
    old_nmds_matrix = nmt.build_nmodes_matrix(old_nmds)
    new_nmds_matrix = nmt.build_nmodes_matrix(new_nmds)
    duszyński = new_nmds_matrix.T @ old_nmds_matrix
    figsize = (6, 6)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, layout='constrained')
    ax.imshow(np.abs(duszyński))
    ax.set_title("Duschinsky matrix")

    tick_positions = [i for i in range(len(old_nmds))]
    old_freqs = [f'{mode.frequency:.0f}' for mode in old_nmds]
    ax.xaxis.set_ticks(ticks=tick_positions, labels=old_freqs, rotation=90)
    ax.yaxis.set_label_text("Old normal modes")

    tick_positions = [i for i in range(len(new_nmds))]
    new_freqs = [f'{mode.frequency:.0f}' for mode in new_nmds]
    ax.yaxis.set_ticks(ticks=tick_positions, labels=new_freqs)
    ax.yaxis.set_label_text("New normal modes")
    plt.show()
    return ax


def main():
    args = get_args()
    old_nmds_path = args.old
    new_nmds_paths = args.new

    old_nmds = nmt.xyz_file_to_NormalModesList(old_nmds_path)
    new_nmds_list = [nmt.xyz_file_to_NormalModesList(path) for path in new_nmds_paths]


    for idx, modes in enumerate(new_nmds_list):
        if args.print_decomposition or args.print_freq_comparison:
            print(f'Normal modes set #{idx}')

        if args.print_freq_comparison is True:
            mode_frequencies(old_nmds, modes)

        if args.print_decomposition is True:
            mode_alignment(old_nmds, modes)

        if args.show_duszynski is True:
            show_duszynski(old_nmds, modes)
        


if __name__ == "__main__":
    main()
