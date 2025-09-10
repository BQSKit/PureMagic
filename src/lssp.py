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
    parser.add_argument("--rseed", "-r", type=int, default=29, help="Random seed")
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
        "--magic-state-lambda",
        "-m",
        type=float,
        # default is 0.00228 with 17 rounds cultivation per timestep
        default=0.0387396,
        help="Lambda parameter for exponential distribution of timesteps to ready a magic state",
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
    circuit = realcircuit.RealCircuit(args)
    circuit.split_ys()
    circuit.check_clifford_relations()
    if "circuit" in args.plot:
        circuit.plot(args.show_product_ids)
    if "freqs" in args.plot:
        circuit.plot_freqs()
    topo_graph = topograph.TopoGraph()
    topo_graph.set_topo(args, circuit.num_qubits, rng)
    if "topo" in args.plot:
        topo_graph.plot(".topo")
    single_scheduler = scheduler.Scheduler(args, 0, 1, rng, topo_graph)
    tot_num_steps, num_scheduled = single_scheduler.schedule_circuit(circuit)
    speedup = float(num_scheduled) / tot_num_steps
    qubit_cost = topo_graph.num_qubits * tot_num_steps
    print(
        f"Scheduled {num_scheduled} in {tot_num_steps} time steps ({speedup:.3f} speedup) "
        f"qubit cost {qubit_cost}"
    )
    num_layers = len(circuit.get_layers())
    speedup = float(num_scheduled) / num_layers
    opt_qubit_cost = topo_graph.num_qubits * num_layers
    print(
        f"Optimal time steps {num_layers} ({speedup:.3f} speedup) " f"qubit cost {opt_qubit_cost}"
    )
    print(f"Scheduling efficiency {(float(opt_qubit_cost) / qubit_cost):.3f}")


if __name__ == "__main__":
    args = get_args()
    main()
