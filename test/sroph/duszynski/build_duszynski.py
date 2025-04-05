import normal_modes_tools as nmt
import normal_modes_tools.compare as compare
import matplotlib.pyplot as plt


def main():
    # Frequency ordered modes
    ground_nmds_path = "../xyz/SrOPh_normal_modes.xyz"
    deuterated_path = "../xyz/SrOPh-5d_normal_modes.xyz"

    # Mulliken ordered modes
    ground_nmds_path = '../mulliken_order/output/SrOPh_nmodes_Mulliken.xyz'
    deuterated_path = '../mulliken_order/output/SrOPh-5d_nmodes_Mulliken.xyz'

    ground_nmds = nmt.xyz_file_to_NormalModesList(ground_nmds_path)
    deuterated_nmds = nmt.xyz_file_to_NormalModesList(deuterated_path)

    modes = [
        {
            'name': r'SrOPh $\tilde{X}$',
            'modes': ground_nmds,
        },
        {
            'name': r'SrOPh-d$_5$ $\tilde{X}$',
            'modes': deuterated_nmds,
        },
    ]

    for mode_item in modes:
        print(f'Normal modes of {mode_item['name']} in the basis of normal'
              ' modes of SrOPh X')
        print("idx  SrOPh X,  SrOPh-d5 X - SrOPh X")
        compare.mode_frequencies(ground_nmds, mode_item['modes'])
        compare.mode_alignment(ground_nmds, mode_item['modes'])
        figsize = (6, 6)
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
        ax = compare.show_duszynski(ground_nmds, mode_item['modes'], ax=ax)
        ax.xaxis.set_label_text(r'Modes of SrOPh $\tilde{X}$')
        ax.yaxis.set_label_text(f'Modes of {mode_item['name']}')
        # fig.savefig('freq_ordered_matrix.pdf')
        fig.savefig('Mulliken_ordered_matrix.pdf')
        


if __name__ == "__main__":
    main()
