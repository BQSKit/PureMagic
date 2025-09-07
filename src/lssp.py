#!/usr/bin/env -S python -u

import sys
import numpy as np
import argparse
import multiprocessing as mp
import topograph
import realcircuit
import scheduler
from utils import timer


def get_args():
    parser = argparse.ArgumentParser(description="Experimental scheduler for the LSSP")
    parser.add_argument(
        "--min-num-qubits", "-n", type=int, default=10, help="Minimum number of data qubits"
    )
    parser.add_argument("--rseed", "-r", type=int, default=29, help="Random seed")
    parser.add_argument(
        "--bus-ratio", "-s", type=int, default=1, help="Ratio of double qubit rows to bus rows"
    )
    plot_options = ["none", "circuit", "paths", "freqs", "topo"]
    parser.add_argument(
        "--plot", "-p", nargs="+", type=str, default="none", choices=plot_options, help="Plotting"
    )
    parser.add_argument(
        "--plot-circuit-range",
        type=str,
        default="",
        help="Min and max depths of circuit to plot: NN:NN",
    )
    parser.add_argument(
        "--circuit", "-c", type=str, required=True, default="None", help="Circuit file name"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--rnd-order", action="store_true", help="Randomly order the qubits")
    parser.add_argument(
        "--magic-steps",
        "-m",
        type=int,
        default=1,
        help="Number of timesteps until a magic state is ready",
    )
    parser.add_argument(
        "--show-product-ids", action="store_true", help="Show product IDs when plotting the circuit"
    )
    parser.add_argument(
        "--log-scheduler", action="store_true", help="Log scheduler actions to .sched file"
    )
    args = parser.parse_args()
    print("Arguments:\n ", "\n  ".join(f"{k}={v}" for k, v in vars(args).items()))
    return args


@timer
def main():
    rng = np.random.default_rng(seed=args.rseed)
    topo_graph = topograph.TopoGraph()
    topo_graph.set_dims(args, rng)
    print("Layout dimensions:", topo_graph.num_cols, topo_graph.num_rows)
    if topo_graph.num_data_qubits != args.min_num_qubits:
        print(
            "Adjusted number of data qubits from",
            args.min_num_qubits,
            "to",
            topo_graph.num_data_qubits,
        )
    if "topo" in args.plot:
        topo_graph.plot(".topo")
        # sys.exit(0)
    circuit = realcircuit.RealCircuit(args)
    circuit.check_clifford_relations()
    if "circuit" in args.plot:
        circuit.plot(args.show_product_ids)
    if "freqs" in args.plot:
        circuit.plot_freqs()
    single_scheduler = scheduler.Scheduler(args, 0, 1, rng, topo_graph)
    tot_num_steps, num_scheduled = single_scheduler.schedule_circuit(circuit)
    speedup = float(num_scheduled) / tot_num_steps
    tot_qubits = (
        topo_graph.num_bus_qubits + topo_graph.num_data_qubits + topo_graph.num_magic_qubits
    )
    qubit_cost = tot_qubits * tot_num_steps
    print(
        f"Scheduled {num_scheduled} in {tot_num_steps} time steps ({speedup:.3f} speedup) qubit cost {qubit_cost}"
    )
    num_layers = len(circuit.get_layers())
    speedup = float(num_scheduled) / num_layers
    print(f"Optimal time steps {num_layers} ({speedup:.3f} speedup)")
    opt_cols, opt_rows = topo_graph.get_topo_dims(1000)
    opt_qubit_cost = num_layers * (opt_cols * opt_rows - int(opt_cols / 2) * 2)
    efficiency = float(opt_qubit_cost) / qubit_cost
    print(f"Optimal qubit cost {opt_qubit_cost}, efficiency {efficiency:.3f}")


if __name__ == "__main__":
    args = get_args()
    main()
