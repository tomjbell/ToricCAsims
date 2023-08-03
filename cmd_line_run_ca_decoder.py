import argparse
import sys
from ca_decoder import lossy_tooms_sweeps
import numpy as np


if __name__ == '__main__':
    # Parse command line input
    parser = argparse.ArgumentParser(description="Pass variables for simulation of high dim toric code")
    parser.add_argument("-dist", "--distances", help="List of code distances", nargs="*", type=int, default=[3],)
    parser.add_argument("-es", "--error_rates", nargs="*", help="list of error rates", type=float, default=[0.],)
    parser.add_argument("-ls", "--loss_rates", nargs="*", help="list of loss rates", type=float, default=[0.],)
    parser.add_argument("-o", "--output_dir", help="description of variable", type=str, default="",)
    parser.add_argument("-ns", "--n_shots", help="description of variable", type=int, default=100,)
    parser.add_argument("-ar", "--arix", help="description of variable", type=int, default=0,)
    parser.add_argument("-dim", "--dimension", help="description of variable", type=int, default=5,)
    parser.add_argument("-nca", "--num_ca_iters", help="description of variable", type=int, default=100,)
    parser.add_argument("-ncaconv", "--num_ca_iters_erasure_conversion", help="description of variable", type=int, default=100,)
    parser.add_argument("-chngf", "--change_dir_freq", help="description of variable", type=int, default=20,)
    parser.add_argument("-qbtdim", "--qubit_cell_dim", help="description of variable", type=int, default=2,)

    try:
        parser.add_argument("-erasconv", "--erasure_conversion", action=argparse.BooleanOptionalAction,)
        parser.add_argument("-show", "--show_fig", action=argparse.BooleanOptionalAction,)
        parser.add_argument("-sf", "--savefig", action=argparse.BooleanOptionalAction, )
        parser.add_argument("-sd", "--savedata", action=argparse.BooleanOptionalAction, )
        parser.add_argument("-fa", "--from_array", action=argparse.BooleanOptionalAction,)
    except AttributeError:
        parser.add_argument("-erasconv", "--erasure_conversion", action='store_true',)
        parser.add_argument("-show", "--show_fig", action='store_true',)
        parser.add_argument("-sf", "--savefig", action='store_true',)
        parser.add_argument("-sd", "--savedata", action='store_true',)
        parser.add_argument("-fa", "--from_array", action='store_true',)

    arguments = parser.parse_args(sys.argv[1:])
    Ls = arguments.distances
    error_rates = arguments.error_rates
    loss_rates = arguments.loss_rates
    output_dir = arguments.output_dir
    savefig = arguments.savefig
    savedata = arguments.savedata
    n_shots = arguments.n_shots
    arix = arguments.arix
    from_array = arguments.from_array
    eras_conv_iters = arguments.num_ca_iters_erasure_conversion
    num_ca_iters = arguments.num_ca_iters
    change_dir_every = arguments.change_dir_freq
    qubit_cell_dim = arguments.qubit_cell_dim
    erasure_conversion = arguments.erasure_conversion
    dimension = arguments.dimension


    if from_array:
        loss_values = np.linspace(0.001, 0.015, 13)
        loss_rates = loss_values[arix]

    lossy_tooms_sweeps(Ls, dim=dimension, error_rates=error_rates, loss=loss_rates, n_shots=n_shots, savefig=savefig, outdir=output_dir,
                       save_data=savedata, n_ca_iters=num_ca_iters, eras_conv_iters=eras_conv_iters, showfig=False,
                       change_dir_every=change_dir_every, qubit_cell_dim=qubit_cell_dim, eras_convert=erasure_conversion)

