import argparse
import sys
from multiprocessing import Pool, cpu_count
from ca_decoder import lossy_tooms_sweeps, tooms_with_loss_parallelized


if __name__ == '__main__':
    # Parse command line input
    parser = argparse.ArgumentParser(description="Pass variables for simulation of high dim toric code")
    parser.add_argument(
        "-dist",
        "--distances",
        help="description of variable",
        nargs="*",
        type=int,
        default=[3],
    )
    parser.add_argument(
        "-es",
        "--error_rates",
        nargs="*",
        help="description of variable",
        type=float,
        default=[0.],
    )
    parser.add_argument(
        "-ls",
        "--loss_rates",
        nargs="*",
        help="description of variable",
        type=float,
        default=[0.],
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="description of variable",
        type=str,
        default="",
    )
    parser.add_argument(
        "-sf",
        "--savefig",
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "-sd",
        "--savedata",
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "-ns",
        "--n_shots",
        help="description of variable",
        type=int,
        default=100,
    )

    arguments = parser.parse_args(sys.argv[1:])
    Ls = arguments.distances
    error_rates = arguments.error_rates
    loss_rates = arguments.loss_rates
    output_dir = arguments.output_dir
    savefig = arguments.savefig
    savedata = arguments.savedata
    n_shots = arguments.n_shots

    # a = tooms_with_loss_parallelized(Ls, error_rate=error_rates, loss_rate=loss_rates, n_shots=n_shots, savefig=savefig, output_dir=output_dir)
    lossy_tooms_sweeps(Ls, error_rates=error_rates, loss=loss_rates, n_shots=n_shots, savefig=savefig, outdir=output_dir, save_data=savedata)

    print()
