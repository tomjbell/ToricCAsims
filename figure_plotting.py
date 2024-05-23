import matplotlib.pyplot as plt
from helpers import load_obj
import seaborn as sns
import os
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



def tooms_5d_no_loss():
    sns.set_theme()
    plt.style.use('seaborn-v0_8')
    distances = [3, 4, 5]
    iter_list = [10, 20, 50, 100]
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), layout='tight')

    for i, num_iters in enumerate(iter_list):
        ax = axs[i//2, i % 2]
        data_dir = '24_05_21'

        plot_helper(data_dir, distances, num_iters, ax, inset_data=num_iters==10)

        ax.set_title(f"{num_iters} iterations")
        ax.set_xlabel('Physical error rate')
        ax.set_ylabel('Logical error rate')
        ax.legend()
    plt.savefig('outputs/Thesis_plots/tooms_5d_no_loss_with_inset', dpi=500)
    plt.show()

def plot_helper(data_dir, Ls, num_iters, axis, inset_data=False):
    path_to_file = os.path.join(os.getcwd(), 'outputs', data_dir)
    for L in Ls:
        filename = f'dim5_Tooms_L_{L}_2000shots_paulicaiters{num_iters}_erasconviters0_'

        x_data = None
        y_data = None
        n_files_opened = 0
        for file in os.listdir(path_to_file):
            if file.startswith(filename):
                n_files_opened += 1
                data = load_obj(path=path_to_file, name=file, suffix='')
                this_x = [k[0] for k in data.keys()]
                if min(this_x) < 0.011:
                    if x_data is None:
                        x_data = [k[0] for k in data.keys()]
                    else:
                        assert x_data == [k[0] for k in data.keys()]
                    if y_data is None:
                        y_data = np.array([v[0] for v in data.values()])
                    else:
                        y_data += np.array([v[0] for v in data.values()])
        if not n_files_opened:
            print(filename, path_to_file)
            print(os.listdir(path_to_file))
        axis.plot(x_data, y_data/n_files_opened, 'o-', label=f"{L=}")
        if inset_data:
            ax_ins = inset_axes(axis, width="45%", height="45%", loc=2)
            data_dir = '24_05_22'
            plot_helper(data_dir, Ls, num_iters, ax_ins, inset_data=False)
            ax_ins.set_yticks([])
            ax_ins.legend()

def gauss_elim_results(dim=3, n_shots=1000, num_points=9):
    plt.style.use('seaborn-v0_8')
    Ls = [3, 5, 7, 9, 11]
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), layout='tight')

    for i in range(2):
        data_dir = '24_05_22/gausselim'
        path_to_file = os.path.join(os.getcwd(), 'outputs', data_dir)
        ax = axs[i]

        filename = f"Gauss_elim_loss_thresh_qubitcell{i+1}_dim{dim}_L{Ls[0]}-{Ls[-1]}_{n_shots}shots_{num_points}points"

        x_data = None
        y_data = None
        n_files_opened = 0

        for file in os.listdir(path_to_file):

            if file.startswith(filename):
                n_files_opened += 1
                data = load_obj(path=path_to_file, name=file, suffix='')
                if x_data is None:
                    x_data = data[0]
                    y_data = {k: np.array(list(v.values())) for k, v in data[1].items()}
                    print(y_data)
                else:
                    for k, v in data[1].items():
                        y_data[k] += np.array(list(v.values()))
                # print(data)
                # exit()
                # this_x = [k[0] for k in data.keys()]
                # if min(this_x) < 0.011:
                #     if x_data is None:
                #         x_data = [k[0] for k in data.keys()]
                #     else:
                #         assert x_data == [k[0] for k in data.keys()]
                #     if y_data is None:
                #         y_data = np.array([v[0] for v in data.values()])
                #     else:
                #         y_data += np.array([v[0] for v in data.values()])
        if not n_files_opened:
            print(filename, path_to_file)
            print(os.listdir(path_to_file))
        else:
            for k, v in y_data.items():
                ax.plot(x_data, v / n_files_opened, 'o-', label=f"L={k}")

            ax.legend()

        ax.set_title(f"{i+1}-cell qubit erasure")
        ax.set_xlabel('Physical loss rate')
        ax.set_ylabel('Logical loss rate')
    plt.savefig('outputs/Thesis_plots/3d_gausselim', dpi=500)
    plt.show()


def sweep_5d_no_loss():
    sns.set_theme()
    plt.style.use('seaborn-v0_8')
    distances = [3, 4, 5]
    iter_list = [10, 20, 50, 100]
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), layout='tight')

    for i, num_iters in enumerate(iter_list):
        ax = axs[i//2, i % 2]
        data_dir = '24_05_22/sweep'

        plot_helper_sweep(data_dir, distances, num_iters, ax, inset_data=False)

        ax.set_title(f"{num_iters} iterations")
        ax.set_xlabel('Physical error rate')
        ax.set_ylabel('Logical error rate')
        ax.legend()
    plt.savefig('outputs/Thesis_plots/sweep_5d_no_loss', dpi=500)
    plt.show()

def plot_helper_sweep(data_dir, Ls, num_iters, axis, inset_data=False):
    path_to_file = os.path.join(os.getcwd(), 'outputs', data_dir)
    filename = f"sweep_rule_L3-5uptoL5_dim5_{num_iters}sweeps_1000shots"
    x_data = np.linspace(0.014, 0.027, 7)
    y_data = None
    n_files_opened = 0
    for file in os.listdir(path_to_file):
        if file.startswith(filename):
            n_files_opened += 1
            data = load_obj(path=path_to_file, name=file, suffix='')
            if y_data is None:
                y_data = {k: np.array(v) for k, v in data.items()}
            else:
                for k, v in data.items():
                    y_data[k] += np.array(v)

    if not n_files_opened:
        print(filename, path_to_file)
        print(os.listdir(path_to_file))
    else:
        for k, v in y_data.items():
            axis.plot(x_data, v/n_files_opened, 'o-', label=f"L={k}")
        if inset_data:
            ax_ins = inset_axes(axis, width="45%", height="45%", loc=2)
            data_dir = '24_05_22'
            plot_helper(data_dir, Ls, num_iters, ax_ins, inset_data=False)
            ax_ins.set_yticks([])
            ax_ins.legend()



if __name__ == '__main__':
    gauss_elim_results()
    exit()
    tooms_5d_no_loss()
